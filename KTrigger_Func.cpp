#include "KTrigger.h"
#include <string>

static void PrintMatrix(PF_FloatMatrix& matrix);

static void MultiMatrix(PF_FloatMatrix* apply_matrix, PF_FloatMatrix* transform_matrix) {
	PF_FloatMatrix temp_matrix;
	AEFX_CLR_STRUCT(temp_matrix);
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			temp_matrix.mat[i][j] = transform_matrix->mat[i][j];
		}
	}

	for (int i = 0;i < 3;i++) {
		for (int j = 0;j < 3;j++) {
			transform_matrix->mat[i][j] = 0;
			for (int k = 0;k < 3;k++) {
				transform_matrix->mat[i][j] += apply_matrix->mat[i][k] * temp_matrix.mat[k][j];
			}
		}
	}

	//PrintMatrix(*transform_matrix);
}

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

static void ApplyAngleMatrix(PF_FpLong angle, PF_FloatMatrix* transform_matrix) {
	PF_FpLong theta = angle * PF_PI / 180.0f;

	PF_FloatMatrix apply_matrix = { {
		{cos(theta),	-sin(theta),	0},
		{sin(theta),	cos(theta),		0},
		{0,				0,				1}
		} 
	};
	
	MultiMatrix(&apply_matrix, transform_matrix);
}

static void ApplyScaleMatrix(PF_FpLong scale, A_Boolean is_flip, PF_FloatMatrix* transform_matrix) {

	PF_FloatMatrix apply_matrix = { {
		{scale,			0,			0},
		{0,				scale,		0},
		{0,				0,			1}
		}
	};

	if (is_flip) {
		apply_matrix.mat[0][0] *= -1;
	}

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
	for (int p = 0; p < 3; p++) {
		for (int q = 0; q < 3; q++) {
			OutputDebugStringA((std::to_string(matrix.mat[p][q]) + ",").c_str());
		}
		OutputDebugStringA("\n");
	}
	OutputDebugStringA("\n");
}
