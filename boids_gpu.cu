#include <SDL2/SDL.h>
#include <random>
#include <iostream>

#define CUDA_GLOBAL_FUNC __global__
#define CUDA_DEVICE_FUNC __device__
#define CUDA_HOST_FUNC __host__
#define CUDA_CONSTANT __constant__

using namespace std;

CUDA_CONSTANT float _max_distance;
CUDA_CONSTANT int _boid_count, _window_width, _window_height;

struct vec2{
    float x,y;
};

CUDA_CONSTANT vec2 _mouse_pos;

struct boid{
    vec2 position, velocity;
};

CUDA_DEVICE_FUNC float get_distance(const vec2 &a ,const vec2 &b){
    float diffx = a.x-b.x;
    float diffy = a.y-b.y;
    return __fsqrt_rn(diffx*diffx+diffy*diffy);
}

CUDA_DEVICE_FUNC float get_magnitude(const vec2 &a){
    return __fsqrt_rn(a.x*a.x+a.y*a.y);
}

CUDA_DEVICE_FUNC vec2 normalize_vec(const vec2 &a){
    float magnitude = get_magnitude(a);
    return {a.x/magnitude,a.y/magnitude};
}

CUDA_GLOBAL_FUNC void compute_acc(boid *boids, vec2 *acc){
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    int stride = blockDim.x*gridDim.x;

    for(int i = index; i < _boid_count; i += stride){
        vec2 out {0.0f,0.0f}, norm_avg = normalize_vec(boids[i].velocity),pos_avg = boids[i].position;
        int count = 1;
        for(int j = 0; j < i; ++j){
            float dist = get_distance(boids[i].position,boids[j].position);
            if(dist < _max_distance){
                ++count;
                vec2 diff {boids[j].position.x-boids[i].position.x,boids[j].position.y-boids[i].position.y};
                vec2 direction = normalize_vec(diff);
                vec2 bdir = normalize_vec(boids[j].velocity);
                norm_avg.x += bdir.x;
                norm_avg.y += bdir.y;
                pos_avg.x += boids[j].position.x;
                pos_avg.y += boids[j].position.y;
                float force = max((30.0-dist)/1000,0.0);

                out.x -= direction.x * force;
                out.y -= direction.y * force;
            }
        }
        for(int j = i+1; j < _boid_count; ++j){
            float dist = get_distance(boids[i].position,boids[j].position);
            if(dist < _max_distance){
                ++count;
                vec2 diff {boids[j].position.x-boids[i].position.x,boids[j].position.y-boids[i].position.y};
                vec2 direction = normalize_vec(diff);
                vec2 bdir = normalize_vec(boids[j].velocity);
                norm_avg.x += bdir.x;
                norm_avg.y += bdir.y;
                pos_avg.x += boids[j].position.x;
                pos_avg.y += boids[j].position.y;
                float force = max((30.0-dist)/1000,0.0);

                out.x -= direction.x * force;
                out.y -= direction.y * force;
            }
        }
        norm_avg.x /= count;
        norm_avg.y /= count;

        pos_avg.x /= count;
        pos_avg.y /= count;

        vec2 diff = {norm_avg.x-boids[i].velocity.x,norm_avg.y-boids[i].velocity.y};
        vec2 norm = normalize_vec(diff);
        out.x += norm.x*0.0016;
        out.y += norm.y*0.0016;

        vec2 pos_diff = {pos_avg.x-boids[i].position.x,pos_avg.y-boids[i].position.y};
        vec2 pos_norm = normalize_vec(pos_diff);

        out.x += norm.x * 0.0016;
        out.y += norm.y * 0.0016;

        float mouse_dist = get_distance(boids[i].position,_mouse_pos);
        if(mouse_dist < _max_distance){
            vec2 mouse_diff = {boids[i].position.x-_mouse_pos.x,boids[i].position.y-_mouse_pos.y};;
            vec2 mouse_norm = normalize_vec(mouse_diff);

            float force = max((50.0-mouse_dist)/500,0.0);
            out.x += mouse_norm.x * force;
            out.y += mouse_norm.y * force;
        }

        if(boids[i].position.x < 80){
            out.x += max(80.0-boids[i].position.x,0.0)/1000.0;
        }else if(boids[i].position.x > _window_width-80){
            out.x -= max(80.0-(_window_width-boids[i].position.x),0.0)/1000.0;
        }   
        
        if(boids[i].position.y < 80){
            out.y += max(80.0-boids[i].position.y,0.0)/1000.0;
        }else if(boids[i].position.y > _window_height-80){
            out.y -= max(80.0-(_window_height-boids[i].position.y),0.0)/1000.0;
        }
        

        acc[i] = out;
    }
}

CUDA_GLOBAL_FUNC void update_boids(boid *boids ,const vec2 *acc){
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    int stride = blockDim.x*gridDim.x;

    for(int i = index; i < _boid_count; i += stride){
        boids[i].velocity.x += acc[i].x;
        boids[i].velocity.y += acc[i].y;
        
        boids[i].position.x += boids[i].velocity.x;
        boids[i].position.y += boids[i].velocity.y;
    }
}

int main(){
    SDL_Init(SDL_INIT_VIDEO);
    
    SDL_DisplayMode display_mode;
    SDL_GetDesktopDisplayMode(0,&display_mode);
    int window_width = display_mode.w ,window_height = display_mode.h;

    SDL_Window *window = SDL_CreateWindow("Boids",SDL_WINDOWPOS_CENTERED,SDL_WINDOWPOS_UNDEFINED,window_width,window_height,SDL_WINDOW_SHOWN|SDL_WINDOW_FULLSCREEN);
    SDL_Renderer *renderer = SDL_CreateRenderer(window,-1,SDL_RENDERER_ACCELERATED);

    SDL_Event event;
    bool quit = false;

    const int fps = 60;
    const int frame_delay = 1000/fps;
    unsigned int current,reference=0;

    const int boid_count = 200;
    const float max_distance = 50.0f;

    cudaMemcpyToSymbol(_max_distance,&max_distance,sizeof(float));
    cudaMemcpyToSymbol(_boid_count,&boid_count,sizeof(int));
    cudaMemcpyToSymbol(_window_width,&window_width,sizeof(int));
    cudaMemcpyToSymbol(_window_height,&window_height,sizeof(int));

    boid *boids;
    cudaMallocManaged(&boids,boid_count*sizeof(boid));

    vec2 *acc;
    cudaMallocManaged(&acc,boid_count*sizeof(vec2));

    random_device rd;
    mt19937_64 engine(rd());
    uniform_int_distribution<int> dist_x(0,window_width), dist_y(0,window_height);
    uniform_real_distribution<float> dist_vel(-1.0f,1.0f);
    for(int i = 0; i < boid_count; ++i){
        boids[i].position = {(float)dist_x(engine),(float)dist_y(engine)};
        boids[i].velocity = {dist_vel(engine),dist_vel(engine)};
    }

    while(!quit){
        while(SDL_PollEvent(&event)){
            if(event.type == SDL_QUIT){
                quit = true;
                break;
            }else if(event.type == SDL_KEYDOWN){
                switch(event.key.keysym.sym){
                    case SDLK_ESCAPE:
                        quit = true;
                        break;
                }
            }else if(event.type == SDL_MOUSEMOTION){
                vec2 pos = {(float)event.motion.x,(float)event.motion.y};
                cudaMemcpyToSymbol(_mouse_pos,&pos,sizeof(vec2));
            }
        }
        current = SDL_GetTicks();
        if(current-reference > frame_delay){
            reference = current;

            SDL_SetRenderDrawColor(renderer,0,0,0,0xFF);
            SDL_RenderClear(renderer);

            SDL_SetRenderDrawColor(renderer,0xFF,0xFF,0xFF,0xFF);

            compute_acc<<<(boid_count+255)/256,256>>>(boids,acc);
            cudaDeviceSynchronize();

            update_boids<<<(boid_count+255)/256,256>>>(boids,acc);
            cudaDeviceSynchronize();
            SDL_Point points[boid_count];
            for(int i = 0; i < boid_count; ++i) points[i] = {(int)boids[i].position.x,(int)boids[i].position.y};
            SDL_RenderDrawPoints(renderer,points,boid_count);

            SDL_RenderPresent(renderer);           
        }
    }

    cudaFree(boids);
    cudaFree(acc);

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}