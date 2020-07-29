//assignment code

kernel void grey(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	int image_size = get_global_size(0)/3;

	//only applies to operation in the first arary prevents below calls going out of bounds
	if (id/image_size == 0 ){
		//y	  =     0.2126R     +       0.7152G             +       0.0722B 
		B[id] = ((A[id]*0.2126) + (A[id+image_size]*0.7152) + (A[id+(image_size*2)]*0.0722));
		B[id + image_size] = B[id];
		B[id + image_size * 2] = B[id];
	}
	
	
}



kernel void hist_simple(global const uchar* A, global int* H) { 
	int id = get_global_id(0);
	
	//assumes that H has been initialised to 0 (the bins start empty i.e at 0)
	int bin_index = A[id];//take value as a bin index
	atomic_inc(&H[bin_index]);//serial operation, not very efficient!
}



//simple exclusive serial scan based on atomic operations - sufficient for small number of elements
kernel void scan_add_atomic(global int* A, global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	for (int i = id+1; i < N; i++)
		atomic_add(&B[i], A[id]);
}


//find way to make the LUT into a uchar type //possibly just make LUT uchar and have it populated here like normal

kernel void LUT(global const int* A, global int* B) {
	int id = get_global_id(0); //this works like the index
	int value = A[id];
	int max = A[255];
	B[id] = (value * 255) / max;

}


kernel void PROJECT(global const uchar* A, global int* LUT , global uchar* B) {
	int id = get_global_id(0); //this works like the index
	
	B[id] = LUT[A[id]];
}










