#include "KTrigger.h"
#include <string>

static void PrintMatrix(PF_FloatMatrix& matrix);

// static void MultiMatrix(PF_FloatMatrix* apply_matrix, PF_FloatMatrix* transform_matrix) {
// 	PF_FloatMatrix temp_matrix;
// 	AEFX_CLR_STRUCT(temp_matrix);
// 	for (int i = 0; i < 3; ++i) {
// 		for (int j = 0; j < 3; ++j) {
// 			temp_matrix.mat[i][j] = transform_matrix->mat[i][j];
// 		}
// 	}

// 	for (int i = 0;i < 3;i++) {
// 		for (int j = 0;j < 3;j++) {
// 			transform_matrix->mat[i][j] = 0;
// 			for (int k = 0;k < 3;k++) {
// 				transform_matrix->mat[i][j] += apply_matrix->mat[i][k] * temp_matrix.mat[k][j];
// 			}
// 		}
// 	}
// }

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

static void MultiMatrix(const PF_FloatMatrix* apply_matrix, PF_FloatMatrix* transform_matrix) {
    PF_FloatMatrix temp_matrix = *transform_matrix; // 直接拷贝整个结构体
    
    // 展开循环以提高性能
    transform_matrix->mat[0][0] = apply_matrix->mat[0][0] * temp_matrix.mat[0][0] 
                                + apply_matrix->mat[0][1] * temp_matrix.mat[1][0] 
                                + apply_matrix->mat[0][2] * temp_matrix.mat[2][0];
    
    transform_matrix->mat[0][1] = apply_matrix->mat[0][0] * temp_matrix.mat[0][1] 
                                + apply_matrix->mat[0][1] * temp_matrix.mat[1][1] 
                                + apply_matrix->mat[0][2] * temp_matrix.mat[2][1];
    
    transform_matrix->mat[0][2] = apply_matrix->mat[0][0] * temp_matrix.mat[0][2] 
                                + apply_matrix->mat[0][1] * temp_matrix.mat[1][2] 
                                + apply_matrix->mat[0][2] * temp_matrix.mat[2][2];
    
    transform_matrix->mat[1][0] = apply_matrix->mat[1][0] * temp_matrix.mat[0][0] 
                                + apply_matrix->mat[1][1] * temp_matrix.mat[1][0] 
                                + apply_matrix->mat[1][2] * temp_matrix.mat[2][0];
    
    transform_matrix->mat[1][1] = apply_matrix->mat[1][0] * temp_matrix.mat[0][1] 
                                + apply_matrix->mat[1][1] * temp_matrix.mat[1][1] 
                                + apply_matrix->mat[1][2] * temp_matrix.mat[2][1];
    
    transform_matrix->mat[1][2] = apply_matrix->mat[1][0] * temp_matrix.mat[0][2] 
                                + apply_matrix->mat[1][1] * temp_matrix.mat[1][2] 
                                + apply_matrix->mat[1][2] * temp_matrix.mat[2][2];
    
    transform_matrix->mat[2][0] = apply_matrix->mat[2][0] * temp_matrix.mat[0][0] 
                                + apply_matrix->mat[2][1] * temp_matrix.mat[1][0] 
                                + apply_matrix->mat[2][2] * temp_matrix.mat[2][0];
    
    transform_matrix->mat[2][1] = apply_matrix->mat[2][0] * temp_matrix.mat[0][1] 
                                + apply_matrix->mat[2][1] * temp_matrix.mat[1][1] 
                                + apply_matrix->mat[2][2] * temp_matrix.mat[2][1];
    
    transform_matrix->mat[2][2] = apply_matrix->mat[2][0] * temp_matrix.mat[0][2] 
                                + apply_matrix->mat[2][1] * temp_matrix.mat[1][2] 
                                + apply_matrix->mat[2][2] * temp_matrix.mat[2][2];
}

// // static void MultiMatrix(const PF_FloatMatrix* a, PF_FloatMatrix* b) {
//     PF_FloatMatrix temp = *b;
//     PF_FloatMatrix result;
//     AEFX_CLR_STRUCT(result);
    
//     for (int i = 0; i < 3; i++) {
//         result.mat[i][0] = a->mat[i][0] * temp.mat[0][0] 
//                          + a->mat[i][1] * temp.mat[1][0] 
//                          + a->mat[i][2] * temp.mat[2][0];
                         
//         result.mat[i][1] = a->mat[i][0] * temp.mat[0][1] 
//                          + a->mat[i][1] * temp.mat[1][1] 
//                          + a->mat[i][2] * temp.mat[2][1];
                         
//         result.mat[i][2] = a->mat[i][0] * temp.mat[0][2] 
//                          + a->mat[i][1] * temp.mat[1][2] 
//                          + a->mat[i][2] * temp.mat[2][2];
//     }
    
//     *b = result;
// }

static void MultiMatrix2D(PF_FloatMatrix* apply_matrix, PF_FloatMatrix* transform_matrix) {
	PF_FloatMatrix temp_matrix;
	AEFX_CLR_STRUCT(temp_matrix);
	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < 2; ++j) {
			temp_matrix.mat[i][j] = transform_matrix->mat[i][j];
		}
	}

	for (int i = 0;i < 2;i++) {
		for (int j = 0;j < 2;j++) {
			transform_matrix->mat[i][j] = 0;
			for (int k = 0;k < 2;k++) {
				transform_matrix->mat[i][j] += apply_matrix->mat[i][k] * temp_matrix.mat[k][j];
			}
		}
	}

}

// static void ApplyAngleMatrix(PF_FpLong angle, PF_FloatMatrix* transform_matrix) {
// 	PF_FpLong theta = angle * PF_PI / 180.0f;

// 	PF_FloatMatrix apply_matrix = { {
// 		{cos(theta),	-sin(theta),	0},
// 		{sin(theta),	cos(theta),		0},
// 		{0,				0,				1}
// 		} 
// 	};
	
// 	MultiMatrix(&apply_matrix, transform_matrix);
// }

static void ApplyAngleMatrix(PF_FpLong angle, PF_FloatMatrix* transform_matrix) {
    // 预计算三角函数值
    const PF_FpLong theta = angle * (PF_PI / 180.0f);
    const PF_FpLong cos_theta = cos(theta);
    const PF_FpLong sin_theta = sin(theta);
    
    // 直接构造旋转矩阵
    const PF_FloatMatrix apply_matrix = { {
        {cos_theta,    -sin_theta,    0},
        {sin_theta,     cos_theta,    0},
        {0,             0,            1}
    }};
    
    MultiMatrix(&apply_matrix, transform_matrix);
}

// static void ApplyScaleMatrix(PF_FpLong scale, A_Boolean is_flip, PF_FloatMatrix* transform_matrix) {

// 	PF_FloatMatrix apply_matrix = { {
// 		{scale,			0,			0},
// 		{0,				scale,		0},
// 		{0,				0,			1}
// 		}
// 	};

// 	if (is_flip) {
// 		apply_matrix.mat[0][0] *= -1;
// 	}

// 	MultiMatrix(&apply_matrix, transform_matrix);
// }

static void ApplyScaleMatrix(PF_FpLong scale, A_Boolean is_flip, PF_FloatMatrix* transform_matrix) {
    const PF_FpLong actual_scale = is_flip ? -scale : scale;
    
    const PF_FloatMatrix apply_matrix = { {
        {actual_scale,  0,            0},
        {0,             scale,        0},
        {0,             0,            1}
    }};
    
    MultiMatrix(&apply_matrix, transform_matrix);
}

static void ApplyTranslateMatrix(PF_FpLong x_offset, PF_FpLong y_offset, PF_FloatMatrix* transform_matrix) {

	PF_FloatMatrix apply_matrix{ {
		{1,			0,			0},
		{0,			1,			0},
		{x_offset,	y_offset,	1}
		}
	};

	MultiMatrix(&apply_matrix, transform_matrix);
}

static void PrintMatrix(PF_FloatMatrix& matrix) {
	// for (int p = 0; p < 3; p++) {
	// 	for (int q = 0; q < 3; q++) {
	// 		OutputDebugStringA((std::to_string(matrix.mat[p][q]) + ",").c_str());
	// 	}
	// 	OutputDebugStringA("\n");
	// }
	// OutputDebugStringA("\n");
}

static void SetRectIntoRectByPoints(PF_Rect* rect, RectByPoints& points) {
    points.topLeft = { rect->left, rect->top };
    points.topRight = { rect->right, rect->top };
    points.bottomLeft = { rect->left, rect->bottom };
    points.bottomRight = { rect->right, rect->bottom };
}

static void MultiMatrixForRectByPoints(const PF_FloatMatrix* matrix, RectByPoints& points) {
    // 把 RectByPoints 的四个点进行矩阵变换
    PF_Point temp;

    // Transform topLeft
    temp.x = static_cast<A_long>(matrix->mat[0][0] * points.topLeft.x + matrix->mat[1][0] * points.topLeft.y + matrix->mat[2][0]);
    temp.y = static_cast<A_long>(matrix->mat[0][1] * points.topLeft.x + matrix->mat[1][1] * points.topLeft.y + matrix->mat[2][1]);
    points.topLeft = temp;

    // Transform topRight
    temp.x = static_cast<A_long>(matrix->mat[0][0] * points.topRight.x + matrix->mat[1][0] * points.topRight.y + matrix->mat[2][0]);
    temp.y = static_cast<A_long>(matrix->mat[0][1] * points.topRight.x + matrix->mat[1][1] * points.topRight.y + matrix->mat[2][1]);
    points.topRight = temp;

    // Transform bottomLeft
    temp.x = static_cast<A_long>(matrix->mat[0][0] * points.bottomLeft.x + matrix->mat[1][0] * points.bottomLeft.y + matrix->mat[2][0]);
    temp.y = static_cast<A_long>(matrix->mat[0][1] * points.bottomLeft.x + matrix->mat[1][1] * points.bottomLeft.y + matrix->mat[2][1]);
    points.bottomLeft = temp;

    // Transform bottomRight
    temp.x = static_cast<A_long>(matrix->mat[0][0] * points.bottomRight.x + matrix->mat[1][0] * points.bottomRight.y + matrix->mat[2][0]);
    temp.y = static_cast<A_long>(matrix->mat[0][1] * points.bottomRight.x + matrix->mat[1][1] * points.bottomRight.y + matrix->mat[2][1]);
    points.bottomRight = temp;
}

static void GetMaxRectWithRectByPoints(const RectByPoints& points, PF_Rect* out_rect) {
    // 根据 RectByPoints 的四个点计算出包含它们的最大矩形，不要用 std
    out_rect->left = min(min(points.topLeft.x, points.bottomLeft.x), min(points.topRight.x, points.bottomRight.x));
    out_rect->top = min(min(points.topLeft.y, points.topRight.y), min(points.bottomLeft.y, points.bottomRight.y));
    out_rect->right = max(max(points.topLeft.x, points.bottomLeft.x), max(points.topRight.x, points.bottomRight.x));
    out_rect->bottom = max(max(points.topLeft.y, points.topRight.y), max(points.bottomLeft.y, points.bottomRight.y));
}