__global__ void cost_aggregate(unsigned int * cost_volume, unsigned int * out_cost_volume,
unsigned int volume_h, unsigned int volume_w, unsigned int volume_d, unsigned int para_P1, unsigned int para_P2)
{

  const int h = volume_h;
  const int w = volume_w;
  const int dmax = volume_d;

  const int P1 = para_P1;
  const int P2 = para_P2;

  const int index_volume = h * w * dmax;

  const int d = threadIdx.x;
  //const int d = threadIdx.x; + d_min;
  const int dir = blockIdx.x;

  int dx = 0;
  int dy = 0;

  int x0 = 0;
  int y0 = 0;

  // int x_ = 0;
  // int y_ = 0;

  // int maxItr = 0;

  switch(dir){

    // left_to_right
    case 0:

      dx = 1;
      x0 = 0;

      for (int y = 0; y < h; y ++){
       out_cost_volume[(x0 + (y * w)) * dmax + d + (dir*index_volume)] = cost_volume[(x0 + (y * w)) * dmax + d];
      }

      for (int x = x0 + dx; x < w; x += dx){

        __syncthreads();

        for (int y = 0; y < h; y ++){



          if (cost_volume[(x + (y * w)) * dmax + d] == 30){
            out_cost_volume[(x + (y * w)) * dmax + d + (dir*index_volume)] = 0;
            continue;

          }else{
            
            float term1 = out_cost_volume[(x - dx + (y * w)) * dmax + d + (dir*index_volume)];

            float term2 = (d == 0) ? term1 : out_cost_volume[(x - dx + (y * w)) * dmax + d - 1 + (dir*index_volume)] + P1;
            float term3 = (d == dmax - 1) ? term1 : out_cost_volume[(x - dx + (y * w)) * dmax + d + 1 + (dir*index_volume)] + P1;

            
            float term4 = out_cost_volume[(x - dx + (y * w)) * dmax + (dir*index_volume)] + P2;
            for (int i = 1; i < dmax; i++){
              float test_term4 = out_cost_volume[(x - dx + (y * w)) * dmax + i + (dir*index_volume)] + P2;
              if (test_term4 < term4)
                  term4 = test_term4;
            }

            
            float minVal = fminf(term1,fminf(term2, fminf(term3,term4)));

            // out_cost_volume[(x + (y * w)) * dmax + d + (dir*index_volume)] = cost_volume[(x + (y * w)) * dmax + d] + minVal - term4 + P2;
            out_cost_volume[(x + (y * w)) * dmax + d + (dir*index_volume)] = cost_volume[(x + (y * w)) * dmax + d] + minVal;
          }
        }
      }
      break;


    case 1:

      dx = -1;
      x0 = w - 1;

      for (int y = 0; y < h; y++){
        out_cost_volume[(x0 + (y * w)) * dmax + d + (dir*index_volume)] = cost_volume[(x0 + (y * w)) * dmax + d];
      }

      for (int x = x0 + dx; x >=0 ; x += dx){
        __syncthreads();

        for (int y = 0; y < h; y++){



          if (cost_volume[(x + (y * w)) * dmax + d] == 30){
            out_cost_volume[(x + (y * w)) * dmax + d + (dir*index_volume)] = 0;
            continue;

          }else{

            float term1 = out_cost_volume[(x - dx + (y * w)) * dmax + d + (dir*index_volume)];

            float term2 = (d == 0) ? term1 : out_cost_volume[(x - dx + (y * w)) * dmax + d - 1 + (dir*index_volume)] + P1;
            float term3 = (d == dmax - 1)? term1 : out_cost_volume[(x - dx + (y * w)) * dmax + d + 1 + (dir*index_volume)] + P1;

            
            float term4 =  out_cost_volume[(x - dx + (y * w)) * dmax + (dir*index_volume)] + P2;
            for (int i = 1; i < dmax; i++){
              float test_term4 = out_cost_volume[(x - dx + (y * w)) * dmax + i + (dir*index_volume)] + P2;
              if (test_term4 < term4)
                  term4 = test_term4;
            }
            float minVal = fminf(term1, fminf(term2, fminf(term3, term4)));
            //out_cost_volume[(x + (y * w)) * dmax + d + (dir*index_volume)] = cost_volume[(x + (y * w)) * dmax + d] + minVal - term4 + P2;
            out_cost_volume[(x + (y * w)) * dmax + d + (dir*index_volume)] = cost_volume[(x + (y * w)) * dmax + d] + minVal;

          }
        }
      }
      break;
    
    case 2:

      dy = 1;
      y0 = 0;

      for (int x = 0; x < w; x++){
        out_cost_volume[(x + (y0 * w)) * dmax + d + (dir*index_volume)] = cost_volume[(x + (y0 * w)) * dmax + d];
      }
      for (int y=y0+dy; y<h; y += dy){

        __syncthreads();

        for (int x=0; x < w; x++){

          
          if (cost_volume[(x + (y * w)) * dmax + d] == 30){
            out_cost_volume[(x + (y * w)) * dmax + d + (dir*index_volume)] = 0;
            continue;
          }else{
            float term1 = out_cost_volume[(x + ((y - dy) * w)) * dmax + d + (dir*index_volume)];

            float term2 = (d == 0) ? term1: out_cost_volume[(x + ((y-dy) * w)) * dmax + d - 1 + (dir*index_volume)] + P1;
            float term3 = (d == dmax - 1) ? term1 : out_cost_volume[(x + ((y-dy) * w)) * dmax + d + 1 + (dir*index_volume)] + P1;

            float term4 = out_cost_volume[(x + ((y - dy)*w)) * dmax + (dir*index_volume)] + P2;
            for (int i = 0; i < dmax; i++){
              float test_term4 =  out_cost_volume[(x + ((y - dy)*w)) * dmax + i + (dir*index_volume)] + P2;
              if(test_term4 < term4)
                term4 = test_term4;
            }
            float minVal = fminf(term1,fminf(term2, fminf(term3, term4)));
            //out_cost_volume[(x + (y * w)) * dmax + d + (dir*index_volume)] = cost_volume[(x + (y * w)) * dmax + d] + minVal - term4 + P2;
            out_cost_volume[(x + (y * w)) * dmax + d + (dir*index_volume)] = cost_volume[(x + (y * w)) * dmax + d] + minVal;
          }

        }
      }
      break;
    // top_to_bottom
    case 3:

      dy = -1;
      y0 = h - 1;

      for (int x = 0; x < w; x++){
        out_cost_volume[(x + (y0 * w)) * dmax + d + (dir*index_volume)] = cost_volume[(x + (y0 * w)) * dmax + d];
      }

      for (int y = y0 + dy; y >= 0; y += dy){
        __syncthreads();

        for (int x = 0; x < w; x++){
          

          if (cost_volume[(x + (y * w)) * dmax + d] == 30){
            out_cost_volume[(x + (y * w)) * dmax + d + (dir*index_volume)] = 0;
            continue;
          }else{
            float term1 = out_cost_volume[(x + ((y - dy) * w)) * dmax + d + (dir*index_volume)];

            float term2 = (d == 0) ? term1: out_cost_volume[(x + ((y - dy) * w)) * dmax + d - 1 + (dir*index_volume)] + P1;
            float term3 = (d == dmax -1) ? term1: out_cost_volume[(x + ((y - dy) * w)) * dmax + d + 1 + (dir*index_volume)] + P1;

            float term4 = out_cost_volume[(x + ((y - dy) * w)) * dmax + (dir*index_volume)] + P2;

            for (int i = 1; i < dmax; i ++){
              float test_term4 = out_cost_volume[(x + ((y - dy) * w)) * dmax + i + (dir*index_volume)] + P2;

              if (test_term4 < term4)
                term4 = test_term4;
            }

            float minVal = fminf(term1, fminf(term2, fminf(term3, term4)));
            //out_cost_volume[(x + (y * w)) * dmax + d + (dir*index_volume)] = cost_volume[(x + (y * w)) * dmax + d] + minVal - term4 + P2;
            out_cost_volume[(x + (y * w)) * dmax + d + (dir*index_volume)] = cost_volume[(x + (y * w)) * dmax + d] + minVal;
          }
        }
      }
      break;
  }
}

__global__ void cost_aggregate_diag(unsigned int * cost_volume, unsigned int * out_cost_volume,
unsigned int volume_h, unsigned int volume_w, unsigned int volume_d, unsigned int para_P1, unsigned int para_P2)
{

  const int h = volume_h;
  const int w = volume_w;
  const int dmax = volume_d;

  const int P1 = para_P1;
  const int P2 = para_P2;

  const int index_volume = h * w * dmax;

  const int d = threadIdx.x;
  //const int d = threadIdx.x + d_min;
  const int dir = blockIdx.x;

  int dx = 0;   
  int dy = 0;

  int x0 = 0;
  int y0 = 0;

  int x_ = 0;
  int y_ = 0;

  int maxItr = 0;

  switch(dir){
    // topleft_to_bottomright

    case 0:
      dx = 1;
      dy = -1;

      x0 = 0;
      y0 = h -1;

      for (int x = x0; x < w; x++){
        out_cost_volume[(x + (y0 * w)) * dmax + d + (dir*index_volume)] = cost_volume[(x + (y0 * w)) * dmax + d];
      }

      for (int y=y0; y>=0; y--){
        out_cost_volume[(x0 + (y0 * w)) * dmax + d + (dir*index_volume)] = cost_volume[(x0 + (y * w)) * dmax + d];
      }

      maxItr = (w >= h) ? h : w;
      y_ = y0;
      x_ = x0;

      for (int itr = 1; itr < maxItr; itr++){

        __syncthreads();
        x_ += dx;
        y_ += dy;

        
        int y = y_;
        for (int x = x_; x < w; x++){

          
          if (cost_volume[(x + (y * w)) * dmax + d] == 30){
            out_cost_volume[(x + (y * w)) * dmax + d + (dir*index_volume)] = 0;
            continue;
          }else{

            float term1 = out_cost_volume[(x - dx + ((y - dy) * w))* dmax + d + (dir*index_volume)];

            float term2 = (d == 0) ? term1 : out_cost_volume[(x - dx + ((y - dy) * w)) * dmax + d - 1 + (dir*index_volume)] + P1;
            float term3 = (d == dmax - 1) ? term1 : out_cost_volume[(x - dx + ((y - dy) * w)) * dmax + d + 1 + (dir*index_volume)] + P1;

            float term4 = out_cost_volume[(x - dx + ((y - dy) * w)) * dmax + (dir*index_volume)] + P2;

            for (int i =1; i < dmax; i++){
              float test_term4 = out_cost_volume[(x - dx + ((y - dy) * w)) * dmax + i + (dir*index_volume)] + P2;
              if(test_term4 < term4)
                term4 = test_term4;
            }
            float minVal = fminf(term1, fminf(term2, fminf(term3, term4)));

            out_cost_volume[(x + (y * w)) * dmax + d + (dir*index_volume)] = cost_volume[(x + (y * w)) * dmax + d] + minVal - term4 + P2;
          }

        }

        
        int x = x_;
        for (int y = y_; y >= 0; y--){
          

          if (cost_volume[(x + (y * w)) * dmax + d] == 30){
            out_cost_volume[(x + (y * w)) * dmax + d + (dir*index_volume)] = 0;
            continue;

          }else{

            float term1 = out_cost_volume[(x - dx + ((y - dy) * w))* dmax + d + (dir*index_volume)];
            float term2 = (d == 0) ? term1 : out_cost_volume[(x - dx + ((y - dy) * w)) * dmax + d - 1 + (dir*index_volume)] + P1;
            float term3 = (d == dmax - 1) ? term1 : out_cost_volume[(x - dx + ((y - dy) * w)) * dmax + d + 1 + (dir*index_volume)] + P1;

            float term4 = out_cost_volume[(x - dx + ((y - dy) * w)) * dmax + (dir*index_volume)] + P2;

            for (int i =1; i < dmax; i++){
              float test_term4 = out_cost_volume[(x - dx + ((y - dy) * w)) * dmax + i + (dir*index_volume)] + P2;
              if(test_term4 < term4)
                term4 = test_term4;
            }

            float minVal = fminf(term1, fminf(term2, fminf(term3, term4)));

            out_cost_volume[(x + (y * w)) * dmax + d + (dir*index_volume)] = cost_volume[(x + (y * w)) * dmax + d] + minVal - term4 + P2;
          }

        }

      }
      break;

    // bottomright_to_topleft
    case 1:

      dx = -1;
      dy = 1;
      x0 = w - 1;
      y0 = 0;

      for (int x = x0; x >= 0; x--){
        out_cost_volume[(x + (y0 * w)) * dmax + d + (dir*index_volume)] = cost_volume[(x + (y0 * w)) * dmax + d];
      }
      for (int y = y0; y < h; y++){
        out_cost_volume[(x0 + (y * w)) * dmax + d + (dir*index_volume)] = cost_volume[(x0 + (y * w)) * dmax + d];
      }

      maxItr = (w >= h)? h:w;
      y_ = y0;
      x_ = x0;

      for (int itr = 1; itr < maxItr; itr++){
        __syncthreads();

        x_ += dx;
        y_ += dy;

        
        int y = y_;
        for (int x = x_; x>=0; x--){

          
          if (cost_volume[(x + (y * w)) * dmax + d] == 30){
            out_cost_volume[(x + (y * w)) * dmax + d + (dir*index_volume)] = 0;
            continue;

          }else{
            float term1 = out_cost_volume[(x - dx + ((y - dy) * w))* dmax + d + (dir*index_volume)];

            float term2 = (d == 0) ? term1 : out_cost_volume[(x - dx + ((y - dy) * w)) * dmax + d - 1 + (dir*index_volume)] + P1;
            float term3 = (d == dmax - 1) ? term1 : out_cost_volume[(x - dx + ((y - dy) * w)) * dmax + d + 1 + (dir*index_volume)] + P1;

            float term4 = out_cost_volume[(x - dx + ((y - dy) * w)) * dmax + (dir*index_volume)] + P2;

            for (int i =1; i < dmax; i++){
              float test_term4 = out_cost_volume[(x - dx + ((y - dy) * w)) * dmax + i + (dir*index_volume)] + P2;
              if(test_term4 < term4)
                term4 = test_term4;
            }
            float minVal = fminf(term1, fminf(term2, fminf(term3, term4)));

            out_cost_volume[(x + (y * w)) * dmax + d + (dir*index_volume)] = cost_volume[(x + (y * w)) * dmax + d] + minVal - term4 + P2;
          }
        }

        
        int x = x_;
        for (int y=y_; y <h; y++){

          
          if (cost_volume[(x + (y * w)) * dmax + d] == 30){
            out_cost_volume[(x + (y * w)) * dmax + d + (dir*index_volume)] = 0;
            continue;

          }else{
            float term1 = out_cost_volume[(x - dx + ((y - dy) * w))* dmax + d + (dir*index_volume)];

            float term2 = (d == 0) ? term1 : out_cost_volume[(x - dx + ((y - dy) * w)) * dmax + d - 1 + (dir*index_volume)] + P1;
            float term3 = (d == dmax - 1) ? term1 : out_cost_volume[(x - dx + ((y - dy) * w)) * dmax + d + 1 + (dir*index_volume)] + P1;

            float term4 = out_cost_volume[(x - dx + ((y - dy) * w)) * dmax + (dir*index_volume)] + P2;

            for (int i =1; i < dmax; i++){
              float test_term4 = out_cost_volume[(x - dx + ((y - dy) * w)) * dmax + i + (dir*index_volume)] + P2;
              if(test_term4 < term4)
                term4 = test_term4;
            }
            float minVal = fminf(term1, fminf(term2, fminf(term3, term4)));

            out_cost_volume[(x + (y * w)) * dmax + d + (dir*index_volume)] = cost_volume[(x + (y * w)) * dmax + d] + minVal - term4 + P2;
          }
        }
      }
      break;

    // bottomleft_to_topright
    case 2:
      dx = 1;
      dy = 1;

      x0 = 0;
      y0 = 0;

      for (int x = x0; x < w; x++){
        out_cost_volume[(x + (y0 * w)) * dmax + d + (dir*index_volume)] = cost_volume[(x + (y0 * w)) * dmax + d];
      }
      for (int y = y0; y < h; y++){
        out_cost_volume[(x0 + (y * w))* dmax + d + (dir*index_volume)] = cost_volume[(x0 + (y * w))* dmax + d];
      }

      maxItr = (w >= h) ? h:w;
      y_ = y0;
      x_ = x0;

      for (int itr = 1; itr < maxItr; itr++){

        __syncthreads();

        x_ += dx;
        y_ += dy;

        
        int y = y_;
        for (int x = x_; x < w; x++){

          
          if (cost_volume[(x + (y * w)) * dmax + d] == 30){
            out_cost_volume[(x + (y * w)) * dmax + d + (dir*index_volume)] = 0;
            continue;

          }else{
            float term1 = out_cost_volume[(x - dx + ((y - dy) * w))* dmax + d + (dir*index_volume)];

            float term2 = (d == 0) ? term1 : out_cost_volume[(x - dx + ((y - dy) * w)) * dmax + d - 1 + (dir*index_volume)] + P1;
            float term3 = (d == dmax - 1) ? term1 : out_cost_volume[(x - dx + ((y - dy) * w)) * dmax + d + 1 + (dir*index_volume)] + P1;

            float term4 = out_cost_volume[(x - dx + ((y - dy) * w)) * dmax + (dir*index_volume)] + P2;

            for (int i =1; i < dmax; i++){
              float test_term4 = out_cost_volume[(x - dx + ((y - dy) * w)) * dmax + i + (dir*index_volume)] + P2;
              if(test_term4 < term4)
                term4 = test_term4;
            }
            float minVal = fminf(term1, fminf(term2, fminf(term3, term4)));
            out_cost_volume[(x + (y * w)) * dmax + d + (dir*index_volume)] = cost_volume[(x + (y * w)) * dmax + d] + minVal - term4 + P2;

          }

        }

        

        int x = x_;
        for (int y = y_; y < h; y++){

          
          if (cost_volume[(x + (y * w)) * dmax + d] == 30){
            out_cost_volume[(x + (y * w)) * dmax + d + (dir*index_volume)] = 0;
            continue;

          }else{
            float term1 = out_cost_volume[(x - dx + ((y - dy) * w))* dmax + d + (dir*index_volume)];

            float term2 = (d == 0) ? term1 : out_cost_volume[(x - dx + ((y - dy) * w)) * dmax + d - 1 + (dir*index_volume)] + P1;
            float term3 = (d == dmax - 1) ? term1 : out_cost_volume[(x - dx + ((y - dy) * w)) * dmax + d + 1 + (dir*index_volume)] + P1;

            float term4 = out_cost_volume[(x - dx + ((y - dy) * w)) * dmax + (dir*index_volume)] + P2;

            for (int i =1; i < dmax; i++){
              float test_term4 = out_cost_volume[(x - dx + ((y - dy) * w)) * dmax + i + (dir*index_volume)] + P2;
              if(test_term4 < term4)
                term4 = test_term4;
            }
            float minVal = fminf(term1, fminf(term2, fminf(term3, term4)));

            out_cost_volume[(x + (y * w)) * dmax + d + (dir*index_volume)] = cost_volume[(x + (y * w)) * dmax + d] + minVal - term4 + P2;
          }
        }
      }
      break;

    // topright_to_bottomleft
    case 3:
      dx = -1;
      dy = -1;

      x0 = w - 1;
      y0 = h - 1;

      for (int x = x0; x < w; x++){
        out_cost_volume[(x + (y0 * w)) * dmax + d + (dir*index_volume)] = cost_volume[(x + (y0 * w)) * dmax + d];
      }

      for (int y = y0; y < h; y++){
        out_cost_volume[(x0 + (y * w))* dmax + d + (dir*index_volume)] = cost_volume[(x0 + (y * w))* dmax + d];
      }

      maxItr = (w>=h)? h:w;
      y_ = y0;
      x_ = x0;

      for (int itr=1; itr < maxItr; itr++){
        __syncthreads();
        x_ += dx;
        y_ += dy;

        int y = y_;
        for (int x=x_; x >= 0; x--){

          
          if (cost_volume[(x + (y * w)) * dmax + d] == 30){
            out_cost_volume[(x + (y * w)) * dmax + d + (dir*index_volume)] = 0;
            continue;

          }else{
            float term1 = out_cost_volume[(x - dx + ((y - dy) * w))* dmax + d + (dir*index_volume)];

            float term2 = (d == 0) ? term1 : out_cost_volume[(x - dx + ((y - dy) * w)) * dmax + d - 1 + (dir*index_volume)] + P1;
            float term3 = (d == dmax - 1) ? term1 : out_cost_volume[(x - dx + ((y - dy) * w)) * dmax + d + 1 + (dir*index_volume)] + P1;

            float term4 = out_cost_volume[(x - dx + ((y - dy) * w)) * dmax + (dir*index_volume)] + P2;

            for (int i =1; i < dmax; i++){
              float test_term4 = out_cost_volume[(x - dx + ((y - dy) * w)) * dmax + i + (dir*index_volume)] + P2;
              if(test_term4 < term4)
                term4 = test_term4;
            }
            float minVal = fminf(term1, fminf(term2, fminf(term3, term4)));

            out_cost_volume[(x + (y * w)) * dmax + d + (dir*index_volume)] = cost_volume[(x + (y * w)) * dmax + d] + minVal - term4 + P2;
          }
        }

        int x = x_;
        for (int y = y_; y >= 0; y--){

          
          if (cost_volume[(x + (y * w)) * dmax + d] == 30){
            out_cost_volume[(x + (y * w)) * dmax + d + (dir*index_volume)] = 0;
            continue;

          }else{
            float term1 = out_cost_volume[(x - dx + ((y - dy) * w))* dmax + d + (dir*index_volume)];

            float term2 = (d == 0) ? term1 : out_cost_volume[(x - dx + ((y - dy) * w)) * dmax + d - 1 + (dir*index_volume)] + P1;
            float term3 = (d == dmax - 1) ? term1 : out_cost_volume[(x - dx + ((y - dy) * w)) * dmax + d + 1 + (dir*index_volume)] + P1;

            float term4 = out_cost_volume[(x - dx + ((y - dy) * w)) * dmax + (dir*index_volume)] + P2;

            for (int i =1; i < dmax; i++){
              float test_term4 = out_cost_volume[(x - dx + ((y - dy) * w)) * dmax + i + (dir*index_volume)] + P2;
              if(test_term4 < term4)
                term4 = test_term4;
            }
            float minVal = fminf(term1, fminf(term2, fminf(term3, term4)));

            out_cost_volume[(x + (y * w)) * dmax + d + (dir*index_volume)] = cost_volume[(x + (y * w)) * dmax + d] + minVal - term4 + P2;
          }

        }
      }
      break;
  }
}