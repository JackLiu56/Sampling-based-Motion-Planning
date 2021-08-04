#include <ros/ros.h> 
#include <sensor_msgs/LaserScan.h> 
#include <ackermann_msgs/AckermannDrive.h>
#include <gazebo_msgs/ModelStates.h>
#include <gazebo_msgs/ModelState.h>
#include <gazebo_msgs/SetModelState.h>
#include "gnn.hpp"
#include <stdio.h>
#include <time.h>
#include <random>
///To show result in picture
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#define drawimg false

const double DOUBLEPI_CONSTANT       = 6.283185307179586232;
const double SINGLEPI_CONSTANT       = 3.14159265358979323846;
double angle = 0.0; 
//double distance = 10.0;
const int dimension=7;
const int dimension2=2;
using namespace cv;
const double dis_thre=0.4;
gnn::graph_nearest_neighbors_t* g;
static geometry_msgs::Pose now_state;
std::string car_name("ackermann_vehicle");
const double car_length = 0.335;
static cv::Mat img;
ackermann_msgs::AckermannDrive updated_ackmsg;
float random( int x){
    std::random_device rd;
    std::mt19937 mt(rd());
    float r=mt()%(x*1000);
    return r/1000.0;}
class state_node_t : public gnn::proximity_node_t
{
public:
    state_node_t(int dim) : proximity_node_t()
    {
        state = new double[dim];
        d = dim;
    }
    virtual ~state_node_t()
    {
        delete state;
    }
    double* state;
    int d;
};
cv::Point field2Image(float x, float y, cv::Mat img){
    int height = img.rows;
    int width  = img.cols;
    int x_     = (x+10)*20;
    int y_     = (y+8)*20;
    cv::Point position;
    position.y = x_; //rows
    position.x = y_;  //cols
    return position;
}

cv::Mat drawredpoint(cv::Mat img,gnn::proximity_node_t* in_s)
{

    cv::Point p_start;
    auto s = (state_node_t*)in_s;
    p_start = field2Image(s->state[0],s->state[1],img);
    cv::circle(img,p_start,4,cv::Scalar(0,0,255)); //red
    return img;
}
cv::Mat drawbluepoint(cv::Mat img,gnn::proximity_node_t* in_s)
{

    cv::Point p_start;
    auto s = (state_node_t*)in_s;
    p_start = field2Image(s->state[0],s->state[1],img);
    cv::circle(img,p_start,4,cv::Scalar(255,0,0)); //red
    return img;
}
cv::Mat drawblackpoint(cv::Mat img,gnn::proximity_node_t* in_s)
{

    cv::Point p_start;
    auto s = (state_node_t*)in_s;
    p_start = field2Image(s->state[0],s->state[1],img);
    cv::circle(img,p_start,2,cv::Scalar(0,0,0)); //red
    return img;
}

void print_node(gnn::proximity_node_t* node)
{
    auto n = (state_node_t*)node;
    for(int i=0;i<n->d-1;++i)
    {
        std::cout<<n->state[i]<<", ";
    }
    std::cout<<n->state[n->d-1]<<std::endl;
}

double distance(gnn::proximity_node_t* in_s, gnn::proximity_node_t* in_t)
{
    auto s = (state_node_t*)in_s;
    auto t = (state_node_t*)in_t;
    double weight_tran=1;
    double weight_rot=0;
    double dis_translation,dis_rotation,sum;
    dis_translation=pow(s->state[0]-t->state[0],2.0)+pow(s->state[1]-t->state[1],2.0)+pow(s->state[2]-t->state[2],2.0);
    dis_translation=sqrt(dis_translation);
    dis_rotation=1-abs(s->state[3]*t->state[3]+s->state[4]*t->state[4]+s->state[5]*t->state[5]+s->state[6]*t->state[6]);
    sum = weight_tran * dis_translation + weight_rot * dis_rotation;
    return sum;
}

double distance2(geometry_msgs::Pose position,gnn::proximity_node_t* next_node)
{
    auto next = (state_node_t*)next_node;
    double dis  =0;
    dis = pow(position.position.x-next->state[0],2.0) + pow(position.position.y-next->state[1],2.0);
    return sqrt(dis);
}
void calvelcmd(geometry_msgs::Pose position,gnn::proximity_node_t* next_node)
{
    auto next = (state_node_t*)next_node;
    double dis = 0;
    double radius=0;
    double angle=0;
    double delta_x = next->state[0]-position.position.x;
    double delta_y = next->state[1]-position.position.y;
    dis = pow(delta_x,2.0)+pow(delta_y,2.0);
    dis= sqrt(dis);
    delta_x =delta_x/dis;
    delta_y =delta_y/dis;
    double stheta = 2*position.orientation.z*position.orientation.w;
    double ctheta = pow(position.orientation.w,2.0)-pow(position.orientation.z,2.0);
    /// it is used to represent the angle from in_s to in_e in queration
    double sinhalf=0;
    double coshalf=0;
    sinhalf= sqrt((1-delta_x)/2);
    if(sinhalf!=0)
        coshalf= delta_y/sinhalf/2;
    else
        coshalf = 1;
    double costheta=0;
    costheta = 2*pow((coshalf*position.orientation.w+sinhalf*position.orientation.z),2.0)-1;
    double sintheta=0;
    sintheta = 2 * (coshalf*position.orientation.w+sinhalf*position.orientation.z) * (coshalf*position.orientation.z-sinhalf*position.orientation.w);
    radius = dis/fabs(sintheta)/2;
    angle = atan(car_length/radius/2);
    if(-delta_x*(stheta-delta_y)-(0-delta_y)*(ctheta-delta_x)<0)
        updated_ackmsg.steering_angle = -angle;
    else
        updated_ackmsg.steering_angle = angle;
    updated_ackmsg.speed = 0.05;
}

double dis2line(gnn::proximity_node_t* in_s, gnn::proximity_node_t* in_t)
{
    auto s = (state_node_t*)in_s;
    auto t = (state_node_t*)in_t;
    int nr;
    unsigned int* neighbors = in_s->get_neighbors(&nr);
    if(nr==1)
    {
        auto e =(state_node_t*)g->get_node_by_index(neighbors[0]);
        double dis,d1,d2,l1,l2,l3=0;
        d1=0;
        d2=0;
        l1=0;
        l2=0;
        l3=0;
        for(int i=0;i<dimension2;i++)
        {
            d1=d1+(t->state[i]-s->state[i])*(e->state[i]-s->state[i]);
            d2=d2+(t->state[i]-e->state[i])*(e->state[i]-s->state[i]);
            l1=l1+pow(t->state[i]-s->state[i],2.0);
            l2=l2+pow(t->state[i]-e->state[i],2.0);
            l3=l3+pow(e->state[i]-s->state[i],2.0);
        }
        l1=sqrt(l1);
        l2=sqrt(l2);
        l3=sqrt(l3);
        if(d1>0&&d2<0)
        {
            if(l1*l1-pow(d1/l3,2.0)<0)
            {
                std::cout<<d1<<"  "<<l1<<"  "<<l3<<std::endl;
                std::cout<<"false::"<<l1*l1-pow(d1/l3,2.0)<<std::endl;
            }
            dis=sqrt(l1*l1-pow(d1/l3,2.0));
        }
        else if(d1>0)
        {
            dis=l2;
        }
        else
        {
            dis=l1;
        }
        return dis;
    }
    else
        return distance(in_s,in_t);
}

void model_states_CB(gazebo_msgs::ModelStates msg)
{

//    std::cout<<msg.name.size()<<std::endl;
    for(int i=0;i<msg.name.size();i++)
    {
        if(msg.name[i]==car_name)
            now_state = msg.pose[i];

    }
}

bool isStateValid(state_node_t* state_)
{
    if(fabs(state_->state[0])>9.5)
        return false;
    else if(state_->state[1]>6.5||state_->state[1]<-7.5)
        return false;
    else if((state_->state[0]<-4.4&&state_->state[0]>-5.0)&&(state_->state[1]<1.3))
        return false;
    else if((state_->state[0]<1.1&&state_->state[0]>0.5)&&(state_->state[1]>-2))
        return false;
    else if((state_->state[0]<5.9&&state_->state[0]>5.3)&&(state_->state[1]>-4.5&&state_->state[1]<3))
        return false;
    else
        return true;
}

bool isStateAccess(gnn::proximity_node_t* in_s, gnn::proximity_node_t* in_e)
{
    auto s = (state_node_t*) in_s;
    auto e = (state_node_t*) in_e;
    double dis = distance(s,e);
    double delta_x = s->state[0]-e->state[0];
    double delta_y = s->state[1]-e->state[1];
    delta_x = delta_x/dis;
    delta_y = delta_y/dis;
    /// it is used to represent the angle from in_s to in_e in queration
    double sinhalf=0;
    double coshalf=0;
    sinhalf= sqrt((1-delta_x)/2);
    if(sinhalf!=0)
         coshalf= delta_y/sinhalf/2;
    else
        coshalf = 1;
    double costheta=0;
    double sintheta = 0;
    double radius=0;
    costheta = 2*pow((coshalf*s->state[6]+sinhalf*s->state[5]),2.0)-1;
    sintheta = 2*(coshalf*s->state[6]+sinhalf*s->state[5]) * (coshalf*s->state[5]+sinhalf*s->state[6]);
    if(sintheta=0)
        return false;
    else
    {
        radius = dis/2/fabs(sintheta);
        if(fabs(atan(car_length/radius/2))>SINGLEPI_CONSTANT/10)
            return false;
        else
            return true;
    }
}

///It is built to sample the start node and the end node. Both of them are steadily so they look like
/// x, y 0.3(the geight from piano's center to its bottom), 0,0,sin(theta2),cos(theta2)(for the pitch and roll are equal with 0)
void randomSteadSample(state_node_t* node)
{
    do{
    ///The transtion component
    node->state[0] = random(20)-10;///from -1 to 10
    node->state[1] = random(15)-8; ///from 0 to 10
    node->state[2] = 0.10;
    ///The rotation component
    double theta2;
    theta2=random(1)*DOUBLEPI_CONSTANT;
    node->state[3]= 0;
    node->state[4]= 0;
    node->state[5]= sin(theta2);
    node->state[6]= cos(theta2);
    }
    while(!isStateValid(node));
}

bool isLocalValid(gnn::proximity_node_t* in_s, gnn::proximity_node_t* in_t)
{
    auto s = (state_node_t*)in_s;
    auto t = (state_node_t*)in_t;
    double dis = distance(in_s,in_t);
    if(dis>2)
        return false;
    int nr_mid;
    bool isValid = true;
    ///the distance between two test points is 0.2;
    nr_mid = int(dis/0.05)+1;
    state_node_t* mid_node = new state_node_t(dimension);
    if(!isStateValid(s)||!isStateValid(t))
        return false;
    for(int i=0;i<nr_mid+1;i++)
    {
        mid_node->state[0] = s->state[0] + i * (t->state[0]-s->state[0])/nr_mid;
        mid_node->state[1] = s->state[1] + i * (t->state[1]-s->state[1])/nr_mid;
        mid_node->state[2] = s->state[2] + i * (t->state[2]-s->state[2])/nr_mid;
        mid_node->state[3] = s->state[3] + i * (t->state[3]-s->state[3])/nr_mid;
        mid_node->state[4] = s->state[4] + i * (t->state[4]-s->state[4])/nr_mid;
        mid_node->state[5] = s->state[5] + i * (t->state[5]-s->state[5])/nr_mid;
        mid_node->state[6] = s->state[6] + i * (t->state[6]-s->state[6])/nr_mid;
        if(!isStateValid(mid_node))
        {
            isValid=false;
            delete mid_node;
            return isValid;
        }
    }
    delete mid_node;
    return isValid;
}

bool adjust_new_node(state_node_t* in_s,state_node_t* in_t)
{
    if(!isStateValid(in_s))
    {
        std::cout<<"wrong in_s node"<<std::endl;
        img=drawbluepoint(img,in_s);
    }
    while((!isStateAccess(in_s,in_t)||!isStateValid(in_t)||!isLocalValid(in_s,in_t))&&distance(in_s,in_t)>0.01)
    {
        for(int i=0;i<dimension;i++)
        {
            in_t->state[i]=in_s->state[i]+(in_t->state[i]-in_s->state[i])*0.5;
        }
        double q_mold;
        q_mold =sqrt(pow(in_t->state[3],2.0)+pow(in_t->state[4],2.0)+pow(in_t->state[5],2.0)+pow(in_t->state[6],2.0));
        in_t->state[3] = in_t->state[3]/q_mold;
        in_t->state[4] = in_t->state[4]/q_mold;
        in_t->state[5] = in_t->state[5]/q_mold;
        in_t->state[6] = in_t->state[6]/q_mold;
    }
    if(isLocalValid(in_s,in_t)&&isStateValid(in_t))
        return true;
    else if(!isLocalValid(in_s,in_t)||!isStateValid(in_t))
    {
//        std::cout<<"false"<<std::endl;
        return false;
    }
}

void add_new_node(gnn::graph_nearest_neighbors_t* graph,int closest_index,gnn::proximity_node_t* node)
{
    auto s = (state_node_t*)graph->get_node_by_index(closest_index);
    auto t = (state_node_t*)node;
    int nr;
    unsigned int* neighbors = graph->get_node_by_index(closest_index)->get_neighbors(&nr);
    if(nr==1)
    {
        auto e =(state_node_t*)g->get_node_by_index(neighbors[0]);
        double d1,d2,l1,l2,l3=0;
        for(int i=0;i<dimension2;i++)
        {
            d1=d1+(t->state[i]-s->state[i])*(e->state[i]-s->state[i]);
            d2=d2+(t->state[i]-e->state[i])*(e->state[i]-s->state[i]);
            l1=l1+pow(t->state[i]-s->state[i],2.0);
            l2=l2+pow(t->state[i]-e->state[i],2.0);
            l3=l3+pow(e->state[i]-s->state[i],2.0);
        }
        l1=sqrt(l1);
        l2=sqrt(l2);
        l3=sqrt(l3);
        if(d1>0&&d2<0)
        {
            state_node_t* pro_node = new state_node_t(dimension);
            for(int i =0;i<dimension;i++)
            {
                pro_node->state[i] = s->state[i]+((e->state[i]-s->state[i])/l3)*(d1/l3);
            }
            if(isStateAccess(graph->get_node_by_index(neighbors[0]),pro_node)&&
                    isStateAccess(pro_node,graph->get_node_by_index(closest_index)))
            {
                graph->add_node(pro_node);
                pro_node->add_neighbor(neighbors[0]);
                graph->get_node_by_index(closest_index)->delete_neighbor(neighbors[0]);
                graph->get_node_by_index(closest_index)->add_neighbor(pro_node->get_index());
                if(adjust_new_node(pro_node,t))
                {
                    graph->add_node(t);
                    t->add_neighbor(pro_node->get_index());
                }
            }
        }
        else if(d1>0)
        {
            if(adjust_new_node(e,t))
            {
                graph->add_node(t);
                t->add_neighbor(neighbors[0]);
            }
        }
        else
        {
            if(adjust_new_node(s,t))
            {
                graph->add_node(t);
                t->add_neighbor(closest_index);
            }
        }
    }
    else
    {
        if(adjust_new_node(s,t))
        {
            graph->add_node(t);
            t->add_neighbor(closest_index);
        }
    }
}


cv::Mat drawline(cv::Mat img,gnn::proximity_node_t* in_s,gnn::proximity_node_t* in_t)
{

    cv::Point p_start;
    auto s = (state_node_t*)in_s;
    p_start = field2Image(s->state[0],s->state[1],img);
    cv::Point p_end;
    auto t = (state_node_t*)in_t;
    p_end = field2Image(t->state[0],t->state[1],img);
    cv::line(img,p_start,p_end,cv::Scalar(0,255,0));
    return img;
}
cv::Mat drawredline(cv::Mat img,gnn::proximity_node_t* in_s,gnn::proximity_node_t* in_t)
{

    cv::Point p_start;
    auto s = (state_node_t*)in_s;
    p_start = field2Image(s->state[0],s->state[1],img);
    cv::Point p_end;
    auto t = (state_node_t*)in_t;
    p_end = field2Image(t->state[0],t->state[1],img);
    cv::line(img,p_start,p_end,cv::Scalar(0,0,255));
    return img;
}
cv::Mat drawpath(cv::Mat Img,gnn::graph_nearest_neighbors_t* graph,gnn::proximity_node_t* e)
{
    int i =graph->get_nr_nodes()-1;
    double total_dis=0;
    while(i!=0)
    {
//        std::cout<<"drawpath"<<std::endl;
        int nr;
        unsigned int* neighbors1 = graph->get_node_by_index(i)->get_neighbors(&nr);
//        std::cout<<"drawpath"<<std::endl;
        if(nr==1)
        {
            Img=drawredline(Img,graph->get_node_by_index(i),graph->get_node_by_index(neighbors1[0]));
            total_dis = total_dis+distance(graph->get_node_by_index(i),graph->get_node_by_index(neighbors1[0]));
            i=neighbors1[0];
        }
        else
        {
            std::cout<<"wrong"<<std::endl;
        }
    }
    std::cout<<"total_dis::"<<total_dis<<std::endl;
    return Img;
}
std::vector<int> findpath(gnn::graph_nearest_neighbors_t* graph,gnn::proximity_node_t* e)
{
    int i =graph->get_nr_nodes()-1;
    std::vector<int> result_index;
    result_index.clear();
    double total_dis=0;
    while(i!=0)
    {
//        std::cout<<"drawpath"<<std::endl;
        result_index.push_back(i);
        int nr;
        unsigned int* neighbors1 = graph->get_node_by_index(i)->get_neighbors(&nr);
//        std::cout<<"drawpath"<<std::endl;
        if(nr==1)
        {
            total_dis = total_dis+distance(graph->get_node_by_index(i),graph->get_node_by_index(neighbors1[0]));
            i=neighbors1[0];
        }
        else
        {
            std::cout<<"wrong"<<std::endl;
            return result_index;
        }

    }
    std::cout<<"total_dis::"<<total_dis<<std::endl;
    result_index.push_back(0);
    return result_index;
}
int main(int argc, char** argv)
{
    ros::init(argc, argv, "avoidObstacle");
    ros::NodeHandle n;
    ros::Publisher ackmsg_pub = n.advertise<ackermann_msgs::AckermannDrive>("/ackermann_cmd", 1000);
    ros::Subscriber modelStates_sub_ = n.subscribe("/gazebo/model_states", 100, model_states_CB);
    ros::Rate loop_rate(30);
    img = imread(argv[1], CV_LOAD_IMAGE_UNCHANGED);
    g = new gnn::graph_nearest_neighbors_t(std::bind(&isLocalValid,std::placeholders::_1,std::placeholders::_2),std::bind(&distance,std::placeholders::_1,std::placeholders::_2),
                                           std::bind(&dis2line,std::placeholders::_1,std::placeholders::_2));
    std::vector<state_node_t*> all_nodes;
    state_node_t* start_node = new state_node_t(dimension);
    state_node_t* end_node = new state_node_t(dimension);
    start_node->state[0] = -9;
    start_node->state[1] = -7;
    start_node->state[2] = 0.3;
    start_node->state[3] =  0;
    start_node->state[4] =  0;
    start_node->state[5] =  0;
    start_node->state[6] =  1;
    end_node->state[0] = 9;
    end_node->state[1] = 6;
    end_node->state[2] = 0.3;
    end_node->state[3] =  0;
    end_node->state[4] =  0;
    end_node->state[5] =  0;
    end_node->state[6] =  1;
    if(isStateValid(start_node))
    {
        g->add_node(start_node);
        all_nodes.push_back(start_node);
    }
    int num=0;
    int closest_index=-1;
    double min_dis = 10000;
    double dis;
    while(g->get_nr_nodes()<500&&
          dis2line(g->get_node_by_index(g->get_nr_nodes()-1),end_node)>dis_thre)
    {
        auto new_node = new state_node_t(dimension);
        randomSteadSample(new_node);
        closest_index=-1;
        dis=0;
        closest_index=g->find_closest(new_node,&dis)->get_index();
        if(closest_index!=-1)
        {
            add_new_node(g,closest_index,new_node);
            num++;
        }
        if(min_dis>dis2line(g->get_node_by_index(g->get_nr_nodes()-1),end_node))
            min_dis=dis2line(g->get_node_by_index(g->get_nr_nodes()-1),end_node);
    }
    std::cout<<"min_dis::"<<min_dis<<std::endl;
    std::cout<<"num of sample "<<num<<std::endl;
    std::cout<<"num of vertex"<<g->get_nr_nodes()<<std::endl;
    closest_index=g->find_closest(end_node,&dis)->get_index();
    if(closest_index!=-1)
    {
        add_new_node(g,closest_index,end_node);
        num++;
    }
//#if drawimg
    for(int i=1;i<g->get_nr_nodes()-2;i++)
        img=drawblackpoint(img,g->get_node_by_index(i));
    img = drawredpoint(img,start_node);
    img = drawbluepoint(img,end_node);
    std::cout<<"drawtree"<<std::endl;
//    for(int i = 1; i < g->get_nr_nodes()-1;i++)
//    {
////        img=drawpoint(img,g->get_node_by_index(i));
//        int nr;
//        unsigned int* neighbors = g->get_node_by_index(i)->get_neighbors(&nr);
//        if(nr==1)
//            img = drawline(img,g->get_node_by_index(i),g->get_node_by_index(neighbors[0]));
//    }
    std::cout<<"drawpath::"<<std::endl;
    drawpath(img,g,end_node);
    imwrite("result_img.png",img);
//    cv::imshow("result",img);
//    waitKey(0);
//    return 0;
//#else
    std::vector<int> path_index;
    path_index = findpath(g,end_node);
    std::cout<<"node of path::"<<path_index.size()<<std::endl;
    int now_node_index = 0;
    path_index.pop_back();
    now_node_index=path_index[path_index.size()-1];
    int count =0;
    print_node(g->get_node_by_index(now_node_index));
    static ros::Time time0 = ros::Time::now();
    static ros::Time time1 = ros::Time::now();
    while(count<40&&n.ok())
    {
        now_node_index = path_index[path_index.size()-1];
        if(distance2(now_state,g->get_node_by_index(now_node_index))>0.3)
        {
            calvelcmd(now_state,g->get_node_by_index(now_node_index));
            ackmsg_pub.publish(updated_ackmsg);
        }
        else
        {
            updated_ackmsg.speed = 0;
            path_index.pop_back();
            print_node(g->get_node_by_index(path_index[path_index.size()-1]));
            ackmsg_pub.publish(updated_ackmsg);
            count++;
            std::cout<<"change"<<count<<std::endl;
        }
        ros::spinOnce();
        loop_rate.sleep();
    }
//#endif
}
