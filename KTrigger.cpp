/*******************************************************************/
/*                                                                 */
/*                      ADOBE CONFIDENTIAL                         */
/*                   _ _ _ _ _ _ _ _ _ _ _ _ _                     */
/*                                                                 */
/* Copyright 2007-2023 Adobe Inc.                                  */
/* All Rights Reserved.                                            */
/*                                                                 */
/* NOTICE:  All information contained herein is, and remains the   */
/* property of Adobe Inc. and its suppliers, if                    */
/* any.  The intellectual and technical concepts contained         */
/* herein are proprietary to Adobe Inc. and its                    */
/* suppliers and may be covered by U.S. and Foreign Patents,       */
/* patents in process, and are protected by trade secret or        */
/* copyright law.  Dissemination of this information or            */
/* reproduction of this material is strictly forbidden unless      */
/* prior written permission is obtained from Adobe Inc.            */
/* Incorporated.                                                   */
/*                                                                 */
/*******************************************************************/

#include "KTrigger.h"
#include <Smart_Utils.h>
#include <list>
#include "KTrigger_Func.cpp"
#include <string>

static AEGP_PluginID* Plugin_ID;

static PF_Err 
About (	
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output )
{
	AEGP_SuiteHandler suites(in_data->pica_basicP);
	
	suites.ANSICallbacksSuite1()->sprintf(	out_data->return_msg,
											"%s v%d.%d\r%s",
											STR(StrID_Name), 
											MAJOR_VERSION,
											MINOR_VERSION,
											STR(StrID_Description));
	return PF_Err_NONE;
}

static PF_Err 
GlobalSetup (	
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output)
{
	AEGP_SuiteHandler	suites(in_data->pica_basicP);

	out_data->my_version = PF_VERSION(	MAJOR_VERSION, 
										MINOR_VERSION,
										BUG_VERSION, 
										STAGE_VERSION, 
										BUILD_VERSION);

	out_data->out_flags |=	PF_OutFlag_NON_PARAM_VARY |
							PF_OutFlag_DEEP_COLOR_AWARE;

	out_data->out_flags2 |= PF_OutFlag2_SUPPORTS_THREADED_RENDERING |
							PF_OutFlag2_SUPPORTS_SMART_RENDER |
							PF_OutFlag2_FLOAT_COLOR_AWARE;
	//out_data->out_flags = PF_OutFlag_DEEP_COLOR_AWARE;
	return PF_Err_NONE;
}

static PF_Err 
ParamsSetup (	
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output )
{
	PF_Err		err		= PF_Err_NONE;
	PF_ParamDef	def;	

	AEFX_CLR_STRUCT(def);

	// 控制器
	PF_ADD_FLOAT_SLIDERX(	STR(StrID_Gain_Param_Name), 
							SKELETON_GAIN_MIN,
							SKELETON_GAIN_MAX,
							-10,
							10,
							SKELETON_GAIN_DFLT,
							PF_Precision_HUNDREDTHS,
							0,
							0,
							GAIN_DISK_ID);

	AEFX_CLR_STRUCT(def);

	// 起始时间
	PF_ADD_FLOAT_SLIDERX(STR(StrID_Start_Time_Param_Name),
		0,
		1000000,
		0,
		100,
		0,
		PF_Precision_HUNDREDTHS,
		0,
		0,
		START_TIME_DISK_ID);

	AEFX_CLR_STRUCT(def);

	// 播放速率
	PF_ADD_FLOAT_SLIDERX(STR(StrID_Playrate_Param_Name),
		-100,
		100,
		0,
		2,
		0,
		PF_Precision_HUNDREDTHS,
		0,
		0,
		PLAYRATE_DISK_ID);

	AEFX_CLR_STRUCT(def);

	// 控制器影响
	PF_ADD_CHECKBOXX(STR(StrID_Is_Control_Param_Name),
		false,
		0,
		IS_CONTROL_DISK_ID);

	AEFX_CLR_STRUCT(def);
	// 冻结已有生成状态
	PF_ADD_CHECKBOXX(STR(StrID_Is_Frozen_Name),
		false,
		0,
		IS_FROZEN_DISK_ID);
	AEFX_CLR_STRUCT(def);
	// 锚点
	PF_ADD_POINT(STR(StrID_Anchor_Param_Name),
		INT2FIX(in_data->width / 2),
		INT2FIX(in_data->height / 2),
		0,
		ANCHOR_DISK_ID
	);
	AEFX_CLR_STRUCT(def);
	// 单独缩放
	PF_ADD_FLOAT_SLIDERX(STR(StrID_Scale_Size_Name),
		0,
		1000,
		0,
		400,
		100,
		PF_Precision_HUNDREDTHS,
		PF_ValueDisplayFlag_PERCENT,
		0,
		SCALE_SIZE_DISK_ID);
	AEFX_CLR_STRUCT(def);
	// 水平翻转
	PF_ADD_CHECKBOXX(STR(StrID_Is_Flip_Param_Name),
		false,
		0,
		IS_FLIP_DISK_ID);

	AEFX_CLR_STRUCT(def);
	// 偏移属性选择
	PF_ADD_POPUP(STR(StrID_Switch_Param_Name),
		2,
		1,
		STR(StrID_Switch_Popup_Choices),
		SWITCH_DISK_ID);
	AEFX_CLR_STRUCT(def);
	// 序列数量
	PF_ADD_FLOAT_SLIDERX(STR(StrID_Offset_Count_Param_Name),
		0,
		10000,
		0,
		10,
		1,
		PF_Precision_INTEGER,
		0,
		0,
		OFFSET_COUNT_DISK_ID);
	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC(STR(StrID_Return_Group_Name), RETURN_GROUP_DISK_ID);
	AEFX_CLR_STRUCT(def);
	// 来回往复
	PF_ADD_CHECKBOXX(STR(StrID_Return_Param_Name),
		false,
		0,
		RETURN_DISK_ID);
	AEFX_CLR_STRUCT(def);
	// 来回往复（不含首尾）
	PF_ADD_CHECKBOXX(STR(StrID_Return2_Param_Name),
		false,
		0,
		RETURN2_DISK_ID);
	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(RETURN_GROUP_END_DISK_ID);
	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC(STR(StrID_Offset_Group_Name), OFFSET_GROUP_ID);
	AEFX_CLR_STRUCT(def);
	// 偏移（点）
	PF_ADD_POINT(STR(StrID_Offset_Param_Name),
		0,
		0,
		0,
		OFFSET_DISK_ID
	);

	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(OFFSET_GROUP_END_ID);
	AEFX_CLR_STRUCT(def);
	PF_ADD_TOPIC(STR(StrID_Rotate_Group_Name), ROTATE_GROUP_ID);
	AEFX_CLR_STRUCT(def);
	// 旋转角度
	PF_ADD_ANGLE(STR(StrID_Rotate_Param_Name),
		0,
		ROTATE_DISK_ID);
	AEFX_CLR_STRUCT(def);
	// 旋转缩放
	PF_ADD_FLOAT_SLIDERX(STR(StrID_Rotate_Scale_Param_Name),
		0,
		1000,
		0,
		400,
		100,
		PF_Precision_HUNDREDTHS,
		PF_ValueDisplayFlag_PERCENT,
		0,
		ROTATE_SCALE_DISK_ID);
	AEFX_CLR_STRUCT(def);
	PF_END_TOPIC(ROTATE_GROUP_END_ID);
	AEFX_CLR_STRUCT(def);
	// 最大时间范围
	PF_ADD_FLOAT_SLIDERX(STR(StrID_Max_Dur_Name),
		0,
		100000,
		0,
		10,
		3,
		PF_Precision_HUNDREDTHS,
		0,
		0,
		MAX_DUR_DISK_ID);

	AEFX_CLR_STRUCT(def);
	// 混合模式
	PF_ADD_POPUP(STR(StrID_Xfer_Mode_Name),
	21,
	1,
	STR(StrID_Xfer_Mode_Choices),
	XFER_MODE_DISK_ID);
	// 第二图层
	AEFX_CLR_STRUCT(def);
	PF_ADD_LAYER(STR(StrID_Second_Layer_Name),
		PF_LayerDefault_NONE,
		SECOND_LAYER_DISK_ID);
	AEFX_CLR_STRUCT(def);
	// 用于刷新用的破滑条
	PF_ADD_FLOAT_SLIDERX(STR(StrID_Random_Name),
		0,
		100,
		0,
		100,
		0,
		PF_Precision_HUNDREDTHS,
		0,
		0,
		RANDOM_DISK_ID);

	out_data->num_params = SKELETON_NUM_PARAMS;

	return err;
}

static void Angle2Matrix(PF_FpLong angle, A_long width, A_long height, PF_FpLong scale, PF_FloatMatrix* transform_matrix) {
	PF_FpLong theta = angle * PF_PI / 180.0f;

	(*transform_matrix).mat[0][0] = cos(theta) * scale;
	(*transform_matrix).mat[0][1] = -sin(theta) * scale;

	(*transform_matrix).mat[1][0] = sin(theta) * scale;
	(*transform_matrix).mat[1][1] = cos(theta) * scale;

	(*transform_matrix).mat[2][2] = 1;

	(*transform_matrix).mat[2][0] = (double)width / 2;
	(*transform_matrix).mat[2][1] = (double)height / 2;
}

static void Scale2Matrix(A_long width, A_long height, PF_FpLong scale, A_Boolean is_flip, PF_FloatMatrix* transform_matrix) {

	if (is_flip) {
		(*transform_matrix).mat[0][0] = -1 * scale;
		(*transform_matrix).mat[2][0] = width * scale;
	}
	else {
		(*transform_matrix).mat[0][0] = 1 * scale;
	}
	
	(*transform_matrix).mat[1][1] = 1 * scale;

	(*transform_matrix).mat[2][2] = 1;
	//(*transform_matrix).mat[2][1] = height;
}

static void Translate2Matrix(PF_FpLong x_offset, PF_FpLong y_offset, PF_FloatMatrix* transform_matrix) {

	(*transform_matrix).mat[0][0] = 1;

	(*transform_matrix).mat[1][1] = 1;

	(*transform_matrix).mat[2][2] = 1;
	(*transform_matrix).mat[2][0] = x_offset;
	(*transform_matrix).mat[2][1] = y_offset;
}


// 混合模式

typedef struct {
	A_u_long	index;
	A_u_long	mode;
} XferTable;

XferTable XferModes[21] = {
	0,	PF_Xfer_IN_FRONT,
	1,	PF_Xfer_BEHIND,
	2,	0,
	3,	PF_Xfer_DISSOLVE,
	4,	PF_Xfer_ADD,
	5,	PF_Xfer_MULTIPLY,
	6,	PF_Xfer_SCREEN,
	7,	PF_Xfer_OVERLAY,
	8,	PF_Xfer_SOFT_LIGHT,
	9,	PF_Xfer_HARD_LIGHT,
	10,	PF_Xfer_DARKEN,
	11,	PF_Xfer_LIGHTEN,
	12,	PF_Xfer_DIFFERENCE,
	13,	PF_Xfer_HUE,
	14,	PF_Xfer_SATURATION,
	15,	PF_Xfer_COLOR,
	16,	PF_Xfer_LUMINOSITY,
	17,	PF_Xfer_ALPHA_ADD,
	18,	PF_Xfer_EXCLUSION,
	19,	PF_Xfer_SUBTRACT,
	20,	PF_Xfer_DIVIDE,
};


int GetXferMode(int strNum)
{
	return XferModes[strNum - 1].mode;
}

static PF_Err 
Render(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output)
{
	PF_Err				err = PF_Err_NONE;
	AEGP_SuiteHandler	suites(in_data->pica_basicP);

	GainInfo			giP;
	AEFX_CLR_STRUCT(giP);

	PF_Boolean			deepB = PF_WORLD_IS_DEEP(output); // 是否为16位
	PF_NewWorldFlags	flags = PF_NewWorldFlag_CLEAR_PIXELS;
	if (deepB) {
		flags |= PF_NewWorldFlag_DEEP_PIXELS;
	}

	// 初始化背景与混合模式

	const PF_Pixel transparent_black = { 0, 0, 0, 0 };
	ERR(PF_FILL(&transparent_black, &output->extent_hint, output));
	PF_CompositeMode composite_mode;
	AEFX_CLR_STRUCT(composite_mode);
	composite_mode.opacity = 255;
	composite_mode.xfer = GetXferMode(params[XFER_MODE_DISK_ID]->u.pd.value);

	// 获取当前时间
	const A_long ctime = in_data->current_time;
	// 获取关键帧总数
	A_long keyframes = 0;
	ERR(suites.ParamUtilsSuite3()->PF_GetKeyframeCount(in_data->effect_ref,
		SKELETON_GAIN,
		&keyframes));
	// 找到在可渲染范围内的第一个帧的索引
	PF_KeyIndex findex = 0;
	PF_Boolean fhasKey = false;
	A_long fkey_time = 0;
	A_u_long fkey_time_scale = 0;
	ERR(suites.ParamUtilsSuite3()->PF_FindKeyframeTime(in_data->effect_ref,
		SKELETON_GAIN,
		ctime,
		in_data->time_scale,
		PF_TimeDir_LESS_THAN_OR_EQUAL,
		&fhasKey,
		&findex,
		&fkey_time,
		&fkey_time_scale
	));
	if(fhasKey){
		PF_FpLong ftime = PF_FABS((ctime - fkey_time) / (double)fkey_time_scale);
		const PF_FpLong fmax_cache = params[MAX_DUR_DISK_ID]->u.fs_d.value;
		while(ftime < fmax_cache){
			if (!(findex > 0)) {
				break;
			}
			findex -= 1;
			ERR(suites.ParamUtilsSuite3()->PF_KeyIndexToTime(
				in_data->effect_ref,
				SKELETON_GAIN,
				findex,
				&fkey_time,
				&fkey_time_scale));
			ftime = PF_FABS((ctime - fkey_time) / (double)fkey_time_scale);
		}
	}
	else {
		findex = 0;
	}

	PF_FpLong anchor_x = FIX_2_FLOAT(params[ANCHOR_DISK_ID]->u.td.x_value);
	PF_FpLong anchor_y = FIX_2_FLOAT(params[ANCHOR_DISK_ID]->u.td.y_value);

	// 从第一帧开始遍历，如果在时间范围内就叠加上去
	for (int i = findex;i < keyframes;i++) {
		A_long key_time = 0;
		A_u_long key_time_scale = 0;
		ERR(suites.ParamUtilsSuite3()->PF_KeyIndexToTime(
			in_data->effect_ref,
			SKELETON_GAIN,
			i,
			&key_time,
			&key_time_scale));
		// 如果遍历到最后一个帧，则直接跳出
		if (ctime < key_time) {
			break;
		}
		// 如果在时间范围外，则忽略
		if (ctime - key_time > in_data->total_time){
			continue;
		}
		// 最大缓存量（秒）
		if (key_time_scale != 0) {
			const PF_FpLong time = PF_FABS((ctime - key_time) / (double)key_time_scale);
			const PF_FpLong max_cache = params[MAX_DUR_DISK_ID]->u.fs_d.value;
			if (time > max_cache) {
				continue;
			}
		}

		PF_FloatMatrix matrix = { {
			{1,0,0},
			{0,1,0},
			{0,0,1}
			}
		};

		ApplyTranslateMatrix(anchor_x, anchor_y, &matrix);

		// 新世界并初始化
		const A_long widthL = in_data->extent_hint.right - in_data->extent_hint.left;
		const A_long heightL = in_data->extent_hint.bottom - in_data->extent_hint.top;

		// 获取开始时间
		PF_ParamDef startTimeDef;
		AEFX_CLR_STRUCT(startTimeDef);
		ERR(PF_CHECKOUT_PARAM(
			in_data,
			START_TIME_DISK_ID,
			key_time,
			in_data->time_step,
			in_data->time_scale,
			&startTimeDef));
		A_long const start_key_time = startTimeDef.u.fs_d.value * key_time_scale;
		PF_CHECKIN_PARAM(in_data, &startTimeDef);

		// 获取播放速率
		PF_ParamDef playrateDef;
		AEFX_CLR_STRUCT(playrateDef);
		ERR(PF_CHECKOUT_PARAM(
			in_data,
			PLAYRATE_DISK_ID,
			key_time,
			in_data->time_step,
			in_data->time_scale,
			&playrateDef));
		PF_FpLong const playrate = playrateDef.u.fs_d.value;
		PF_CHECKIN_PARAM(in_data, &playrateDef);

		PF_FpLong const pull_frame_time = (ctime - key_time) * playrate + start_key_time;
		if (pull_frame_time < 0) {
			continue;
		}

		// 获取相应时间画面，如果是当前帧则用当前画面
		// 判断是否为当前帧
		bool is_now = true;
		PF_ParamDef imageDef;
		AEFX_CLR_STRUCT(imageDef);
		ERR(PF_CHECKOUT_PARAM(
			in_data,
			SECOND_LAYER_DISK_ID,
			pull_frame_time,
			in_data->time_step,
			in_data->time_scale,
			&imageDef));
		if (imageDef.u.ld.data) {
			if (i + 1 < keyframes) {
				A_long next_key_time = 0;
				A_u_long next_key_time_scale = 0;
				ERR(suites.ParamUtilsSuite3()->PF_KeyIndexToTime(
					in_data->effect_ref,
					SKELETON_GAIN,
					i + 1,
					&next_key_time,
					&next_key_time_scale));
				if (ctime >= next_key_time) {
					is_now = false;
				}
			}
		}
		if (is_now) {
			ERR(PF_CHECKIN_PARAM(in_data, &imageDef));
			// AEFX_CLR_STRUCT(imageDef);
			ERR(PF_CHECKOUT_PARAM(
				in_data,
				PF_Param_LAYER,
				pull_frame_time,
				in_data->time_step,
				in_data->time_scale,
				&imageDef));
		}
		
		// 给 cworld 赋值，水平翻转
		A_Boolean is_flip = false;
		if (params[IS_FLIP_DISK_ID]->u.bd.value && i % 2 == 1) {
			is_flip = true;
		}
		PF_FpLong scale_size = 1;
		if (params[IS_FROZEN_DISK_ID]->u.bd.value) {
			PF_ParamDef currentDef;
			AEFX_CLR_STRUCT(currentDef);
			ERR(PF_CHECKOUT_PARAM(
				in_data,
				SCALE_SIZE_DISK_ID,
				key_time,
				in_data->time_step,
				in_data->time_scale,
				&currentDef));
			scale_size = currentDef.u.fs_d.value / 100;
			ERR(PF_CHECKIN_PARAM(
				in_data, &currentDef));
		} else {
			scale_size = params[SCALE_SIZE_DISK_ID]->u.fs_d.value / 100;
		}
		ApplyScaleMatrix(
			scale_size,
			is_flip,
			&matrix);
		//else {
		//	ERR(PF_COPY(&imageDef.u.ld, &cworld, NULL, NULL));
		//}
		// 获取当前帧数据（如果启用，否则为索引）
		PF_FpLong state = 0;
		const A_long offset_count = params[OFFSET_COUNT_DISK_ID]->u.fs_d.value;
		if (params[IS_CONTROL_DISK_ID]->u.bd.value) {
			PF_ParamDef ckey;
			AEFX_CLR_STRUCT(ckey);
			ERR(suites.ParamUtilsSuite3()->PF_CheckoutKeyframe(in_data->effect_ref,
				SKELETON_GAIN,
				i,
				NULL,
				NULL,
				&ckey));
			state = ckey.u.fs_d.value;
			ERR(suites.ParamUtilsSuite3()->PF_CheckinKeyframe(
					in_data->effect_ref,
					&ckey));
		}
		else {
			if (offset_count > 0) {
				if (params[RETURN_DISK_ID]->u.bd.value && offset_count > 1) {
					if (params[RETURN2_DISK_ID]->u.bd.value) {
						const int mid = i % (offset_count * 2 - 2);
						if (mid < offset_count) {
							state = mid;
						}
						else {
							state = 2 * offset_count - 2 - mid;
						}
					} else {
						const int mid = i % (offset_count * 2);
						if (mid < offset_count) {
							state = mid;
						}
						else {
							state = 2 * offset_count - mid - 1;
						}
					}
				}
				else {
					state = i % offset_count;
				}
			}
			else {
				state = i;
			}
			
		}
		// 混合进新画面中
		switch (params[SWITCH_DISK_ID]->u.pd.value) {
		case 1: {
			PF_FpLong x_offset = 0;
			PF_FpLong y_offset = 0;
			if (params[IS_FROZEN_DISK_ID]->u.bd.value) {
				PF_ParamDef currentDef;
				AEFX_CLR_STRUCT(currentDef);
				ERR(PF_CHECKOUT_PARAM(
					in_data,
					OFFSET_DISK_ID,
					key_time,
					in_data->time_step,
					in_data->time_scale,
					&currentDef));
				x_offset = FIX_2_FLOAT(currentDef.u.td.x_value);
				y_offset = FIX_2_FLOAT(currentDef.u.td.y_value);
				ERR(PF_CHECKIN_PARAM(
					in_data, &currentDef
				));
			}
			else {
				x_offset = FIX_2_FLOAT(params[OFFSET_DISK_ID]->u.td.x_value);
				y_offset = FIX_2_FLOAT(params[OFFSET_DISK_ID]->u.td.y_value);
			}

			ApplyTranslateMatrix(x_offset * state, y_offset * state, &matrix);

			break;
		}
		case 2: {
			PF_FpLong angle = 0;
			if (params[IS_FROZEN_DISK_ID]->u.bd.value) {
				PF_ParamDef currentDef;
				AEFX_CLR_STRUCT(currentDef);
				ERR(PF_CHECKOUT_PARAM(
					in_data,
					ROTATE_DISK_ID,
					key_time,
					in_data->time_step,
					in_data->time_scale,
					&currentDef));
				angle = FIX_2_FLOAT(currentDef.u.ad.value) * state;
				ERR(PF_CHECKIN_PARAM(
					in_data, &currentDef
				));
			}
			else {
				angle = FIX_2_FLOAT(params[ROTATE_DISK_ID]->u.ad.value) * state;
			}
			ApplyAngleMatrix(angle, &matrix);
			ApplyScaleMatrix(params[ROTATE_SCALE_DISK_ID]->u.fs_d.value / 100, false, &matrix);
			break;
			}
		default: {
			break;
			}
		}
		ApplyTranslateMatrix(-anchor_x, -anchor_y, &matrix);
		ERR(in_data->utils->transform_world(
			in_data->effect_ref,
			in_data->quality,
			in_data->in_flags,
			in_data->field,
			&imageDef.u.ld,
			&composite_mode,
			NULL,
			&matrix,
			1,
			true,
			&in_data->extent_hint,
			output));
		ERR(PF_CHECKIN_PARAM(in_data, &imageDef));
	}

	return err;
}

// SmartRender 部分

static PF_Err
PreRender(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_PreRenderExtra* extra)
{
	PF_Err err = PF_Err_NONE;
	PF_RenderRequest req = extra->input->output_request;
	PF_CheckoutResult in_result;

	AEGP_SuiteHandler suites(in_data->pica_basicP);

	PF_Handle	infoH = suites.HandleSuite1()->host_new_handle(sizeof(LayerPack));

	if (infoH) {

		LayerPack* infoP = reinterpret_cast<LayerPack*>(suites.HandleSuite1()->host_lock_handle(infoH));

		if (infoP) {
			extra->output->pre_render_data = infoH;

			if (!err) {
				req.preserve_rgb_of_zero_alpha = TRUE;	//	Hey, we care about zero alpha
				req.field = PF_Field_FRAME;				//	We want checkout_layer to provide a complete frame for sampling

				ERR(extra->cb->checkout_layer(in_data->effect_ref,
					SKELETON_INPUT,
					SKELETON_INPUT,
					&req,
					in_data->current_time,
					in_data->time_step,
					in_data->time_scale,
					&in_result));

				if (!err) {
					AEFX_CLR_STRUCT(*infoP);

					UnionLRect(&in_result.result_rect, &extra->output->result_rect);
					UnionLRect(&in_result.max_result_rect, &extra->output->max_result_rect);
					
					const A_long Lwidth = in_result.max_result_rect.right - in_result.max_result_rect.left,
					             Lheight = in_result.max_result_rect.bottom - in_result.max_result_rect.top;

					//// 计算 LayerPack 部分
					LayerPack layerPack{
						0,
						std::vector<LayerInfo>{}
					};
					// 获取当前时间
					const A_long ctime = in_data->current_time;

					// 获取参数
					PF_ParamDef isflipDef, isfrozenDef, maxdurDef, offsetcountDef, isconDef,
						returnDef, return2Def, switchDef, rotatescaleDef, xferDef, anchorDef;
					AEFX_CLR_STRUCT(isflipDef);
					AEFX_CLR_STRUCT(isfrozenDef);
					AEFX_CLR_STRUCT(maxdurDef);
					AEFX_CLR_STRUCT(offsetcountDef);
					AEFX_CLR_STRUCT(returnDef);
					AEFX_CLR_STRUCT(return2Def);
					AEFX_CLR_STRUCT(switchDef);
					AEFX_CLR_STRUCT(rotatescaleDef);
					AEFX_CLR_STRUCT(xferDef);
					AEFX_CLR_STRUCT(anchorDef);
					ERR(PF_CHECKOUT_PARAM(in_data,
						IS_FLIP_DISK_ID,
						ctime,
						in_data->time_step,
						in_data->time_scale,
						&isflipDef));
					ERR(PF_CHECKOUT_PARAM(in_data,
						IS_FROZEN_DISK_ID,
						ctime,
						in_data->time_step,
						in_data->time_scale,
						&isfrozenDef));
					ERR(PF_CHECKOUT_PARAM(in_data,
						MAX_DUR_DISK_ID,
						ctime,
						in_data->time_step,
						in_data->time_scale,
						&maxdurDef));
					ERR(PF_CHECKOUT_PARAM(in_data,
						OFFSET_COUNT_DISK_ID,
						ctime,
						in_data->time_step,
						in_data->time_scale,
						&offsetcountDef));
					ERR(PF_CHECKOUT_PARAM(in_data,
						IS_CONTROL_DISK_ID,
						ctime,
						in_data->time_step,
						in_data->time_scale,
						&isconDef));
					ERR(PF_CHECKOUT_PARAM(in_data,
						RETURN_DISK_ID,
						ctime,
						in_data->time_step,
						in_data->time_scale,
						&returnDef));
					ERR(PF_CHECKOUT_PARAM(in_data,
						RETURN2_DISK_ID,
						ctime,
						in_data->time_step,
						in_data->time_scale,
						&return2Def));
					ERR(PF_CHECKOUT_PARAM(in_data,
						SWITCH_DISK_ID,
						ctime,
						in_data->time_step,
						in_data->time_scale,
						&switchDef));
					ERR(PF_CHECKOUT_PARAM(in_data,
						ROTATE_SCALE_DISK_ID,
						ctime,
						in_data->time_step,
						in_data->time_scale,
						&rotatescaleDef));
					ERR(PF_CHECKOUT_PARAM(in_data,
						XFER_MODE_DISK_ID,
						ctime,
						in_data->time_step,
						in_data->time_scale,
						&xferDef));
					ERR(PF_CHECKOUT_PARAM(in_data,
						ANCHOR_DISK_ID,
						ctime,
						in_data->time_step,
						in_data->time_scale,
						&anchorDef));

					PF_FpLong anchor_x = FIX_2_FLOAT(anchorDef.u.td.x_value);
					PF_FpLong anchor_y = FIX_2_FLOAT(anchorDef.u.td.y_value);

					layerPack.xfer = xferDef.u.pd.value;

					// 获取关键帧总数
					A_long keyframes = 0;
					ERR(suites.ParamUtilsSuite3()->PF_GetKeyframeCount(in_data->effect_ref,
						SKELETON_GAIN,
						&keyframes));
					// 找到在可渲染范围内的第一个帧的索引
					PF_KeyIndex findex = 0;
					PF_Boolean fhasKey = false;
					A_long fkey_time = 0;
					A_u_long fkey_time_scale = 0;
					ERR(suites.ParamUtilsSuite3()->PF_FindKeyframeTime(in_data->effect_ref,
						SKELETON_GAIN,
						ctime,
						in_data->time_scale,
						PF_TimeDir_LESS_THAN_OR_EQUAL,
						&fhasKey,
						&findex,
						&fkey_time,
						&fkey_time_scale
					));
					if (fhasKey) {
						PF_FpLong ftime = PF_FABS((ctime - fkey_time) / (double)fkey_time_scale);
						PF_ParamDef tempDef;
						AEFX_CLR_STRUCT(tempDef);
						ERR(PF_CHECKOUT_PARAM(in_data,
							MAX_DUR_DISK_ID,
							ctime,
							in_data->time_step,
							in_data->time_scale,
							&tempDef));
						const PF_FpLong fmax_cache = tempDef.u.fs_d.value;
						while (ftime < fmax_cache) {
							if (!(findex > 0)) {
								break;
							}
							findex -= 1;
							ERR(suites.ParamUtilsSuite3()->PF_KeyIndexToTime(
								in_data->effect_ref,
								SKELETON_GAIN,
								findex,
								&fkey_time,
								&fkey_time_scale));
							ftime = PF_FABS((ctime - fkey_time) / (double)fkey_time_scale);
						}
					}
					else {
						findex = 0;
					}
					//// 帧总数
					A_long storage_num = 0;
					// 从第一帧开始遍历，如果在时间范围内就叠加上去
					for (int i = findex;i < keyframes;i++) {
						A_long key_time = 0;
						A_u_long key_time_scale = 0;
						ERR(suites.ParamUtilsSuite3()->PF_KeyIndexToTime(
							in_data->effect_ref,
							SKELETON_GAIN,
							i,
							&key_time,
							&key_time_scale));
						// 如果遍历到最后一个帧，则直接跳出
						if (ctime < key_time) {
							break;
						}
						// 如果在时间范围外，则忽略
						if (ctime - key_time > in_data->total_time) {
							continue;
						}
						// 最大缓存量（秒）
						if (key_time_scale != 0) {
							const PF_FpLong time = PF_FABS((ctime - key_time) / (double)key_time_scale);
							const PF_FpLong max_cache = maxdurDef.u.fs_d.value;
							if (time > max_cache) {
								continue;
							}
						}

						storage_num++;
						LayerInfo storageInfo;
						AEFX_CLR_STRUCT(storageInfo);
						storageInfo.idL = storage_num;

						PF_FloatMatrix matrix = { {
							{1,0,0},
							{0,1,0},
							{0,0,1}
							} 
						};

						ApplyTranslateMatrix(anchor_x, anchor_y, &matrix);

						// 获取开始时间
						PF_ParamDef startTimeDef;
						AEFX_CLR_STRUCT(startTimeDef);
						ERR(PF_CHECKOUT_PARAM(
							in_data,
							START_TIME_DISK_ID,
							key_time,
							in_data->time_step,
							in_data->time_scale,
							&startTimeDef));
						A_long const start_key_time = startTimeDef.u.fs_d.value * key_time_scale;

						// 获取播放速率
						PF_ParamDef playrateDef;
						AEFX_CLR_STRUCT(playrateDef);
						ERR(PF_CHECKOUT_PARAM(
							in_data,
							PLAYRATE_DISK_ID,
							key_time,
							in_data->time_step,
							in_data->time_scale,
							&playrateDef));
						PF_FpLong const playrate = playrateDef.u.fs_d.value;

						PF_FpLong const pull_frame_time = (ctime - key_time) * playrate + start_key_time;
						if (pull_frame_time < 0) {
							continue;
						}

						// 获取相应时间画面，如果是当前帧则用当前画面
						// 判断是否为当前帧
						bool is_now = true;
						PF_ParamDef imageDef;
						AEFX_CLR_STRUCT(imageDef);
						ERR(PF_CHECKOUT_PARAM(
							in_data,
							SECOND_LAYER_DISK_ID,
							pull_frame_time,
							in_data->time_step,
							in_data->time_scale,
							&imageDef));
						if (imageDef.u.ld.data) {
							if (i + 1 < keyframes) {
								A_long next_key_time = 0;
								A_u_long next_key_time_scale = 0;
								ERR(suites.ParamUtilsSuite3()->PF_KeyIndexToTime(
									in_data->effect_ref,
									SKELETON_GAIN,
									i + 1,
									&next_key_time,
									&next_key_time_scale));
								if (ctime >= next_key_time) {
									is_now = false;
								}
							}
						}

						PF_CheckoutResult result;
						AEFX_CLR_STRUCT(result);
						if (is_now) {
							// AEFX_CLR_STRUCT(imageDef);
							extra->cb->checkout_layer(
								in_data->effect_ref,
								PF_Param_LAYER,
								storage_num,
								&req,
								pull_frame_time,
								in_data->time_step,
								in_data->time_scale,
								&result
							);
						} else {
							extra->cb->checkout_layer(
								in_data->effect_ref,
								SECOND_LAYER_DISK_ID,
								storage_num,
								&req,
								pull_frame_time,
								in_data->time_step,
								in_data->time_scale,
								&result
							);
						}

						// 给 cworld 赋值，水平翻转
						A_Boolean is_flip = false;
						if (isflipDef.u.bd.value && i % 2 == 1) {
							is_flip = true;
						}
						PF_FpLong scale_size = 1;
						if (isfrozenDef.u.bd.value) {
							PF_ParamDef currentDef;
							AEFX_CLR_STRUCT(currentDef);
							ERR(PF_CHECKOUT_PARAM(
								in_data,
								SCALE_SIZE_DISK_ID,
								key_time,
								in_data->time_step,
								in_data->time_scale,
								&currentDef));
							scale_size = currentDef.u.fs_d.value / 100;
						}
						else {
							PF_ParamDef currentDef;
							AEFX_CLR_STRUCT(currentDef);
							ERR(PF_CHECKOUT_PARAM(
								in_data,
								SCALE_SIZE_DISK_ID,
								ctime,
								in_data->time_step,
								in_data->time_scale,
								&currentDef));
							scale_size = currentDef.u.fs_d.value / 100;
						}
						ApplyScaleMatrix(
							scale_size,
							is_flip,
							&matrix);
						// else {
						//	ERR(PF_COPY(&imageDef.u.ld, &cworld, NULL, NULL));
						// }
						// 获取当前帧数据（如果启用，否则为索引）
						PF_FpLong state = 0;
						const A_long offset_count = offsetcountDef.u.fs_d.value;
						if (isconDef.u.bd.value) {
							PF_ParamDef ckey;
							AEFX_CLR_STRUCT(ckey);
							ERR(suites.ParamUtilsSuite3()->PF_CheckoutKeyframe(in_data->effect_ref,
								SKELETON_GAIN,
								i,
								NULL,
								NULL,
								&ckey));
							state = ckey.u.fs_d.value;
							ERR(suites.ParamUtilsSuite3()->PF_CheckinKeyframe(
								in_data->effect_ref,
								&ckey));
						}
						else {
							if (offset_count > 0) {
								if (returnDef.u.bd.value) {
									if (return2Def.u.bd.value) {
										const int mid = i % (offset_count * 2 - 2);
										if (mid < offset_count) {
											state = mid;
										}
										else {
											state = 2 * offset_count - 2 - mid;
										}
									}
									else {
										const int mid = i % (offset_count * 2);
										if (mid < offset_count) {
											state = mid;
										}
										else {
											state = 2 * offset_count - mid - 1;
										}
									}
								}
								else {
									state = i % offset_count;
								}
							}
							else {
								state = i;
							}

						}
						// 混合进新画面中
						A_long check_time = ctime;
						if (isfrozenDef.u.bd.value) {
							check_time = key_time;
						}
						switch (switchDef.u.pd.value) {
						case 1: {
							PF_FpLong x_offset = 0;
							PF_FpLong y_offset = 0;
							PF_ParamDef currentDef;
							AEFX_CLR_STRUCT(currentDef);
							ERR(PF_CHECKOUT_PARAM(
								in_data,
								OFFSET_DISK_ID,
								check_time,
								in_data->time_step,
								in_data->time_scale,
								&currentDef));
							x_offset = FIX_2_FLOAT(currentDef.u.td.x_value) * state;
							y_offset = FIX_2_FLOAT(currentDef.u.td.y_value) * state;

							ApplyTranslateMatrix(x_offset, y_offset, &matrix);
							break;
						}
						case 2: {
							PF_FpLong angle = 0;
							PF_ParamDef currentDef;
							AEFX_CLR_STRUCT(currentDef);
							ERR(PF_CHECKOUT_PARAM(
								in_data,
								ROTATE_DISK_ID,
								check_time,
								in_data->time_step,
								in_data->time_scale,
								&currentDef));
							angle = FIX_2_FLOAT(currentDef.u.ad.value) * state;
							ApplyAngleMatrix(angle, &matrix);
							ApplyScaleMatrix(rotatescaleDef.u.fs_d.value / 100, false, &matrix);
							break;
						}
						default: {
							break;
							}
						}

						ApplyTranslateMatrix(-anchor_x, -anchor_y, &matrix);

						storageInfo.matrix = matrix;

						layerPack.pack.push_back(storageInfo);
					}
					// suites.HandleSuite1()->host_resize_handle(sizeof(layerPack),&infoH);
					infoP->xfer = layerPack.xfer;
					infoP->pack = layerPack.pack;
				}
			}

			suites.HandleSuite1()->host_unlock_handle(infoH);
		}
		else {
			err = PF_Err_OUT_OF_MEMORY;
		}
	}
	else {
		err = PF_Err_OUT_OF_MEMORY;
	}
	return err;

}

static PF_Err
SmartRender(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_SmartRenderExtra* extra)

{
	PF_Err				err = PF_Err_NONE,
		err2 = PF_Err_NONE;

	AEGP_SuiteHandler 	suites(in_data->pica_basicP);
	PF_EffectWorld* input_worldP = NULL,
		* output_worldP = NULL,
		* middle_worldP = NULL;
	PF_WorldSuite2* wsP = NULL;
	PF_PixelFormat		format = PF_PixelFormat_INVALID;

	LayerPack* infoP = reinterpret_cast<LayerPack*>(suites.HandleSuite1()->host_lock_handle(reinterpret_cast<PF_Handle>(extra->input->pre_render_data)));

	if (infoP) {
		ERR((extra->cb->checkout_layer_pixels(in_data->effect_ref, SKELETON_INPUT, &input_worldP)));

		ERR(extra->cb->checkout_output(in_data->effect_ref, &output_worldP));

		if (!err && output_worldP) {

			const PF_Pixel transparent_black = { 0, 0, 0, 0 };
			ERR(PF_FILL(&transparent_black, &output_worldP->extent_hint, output_worldP));
			PF_CompositeMode composite_mode;
			AEFX_CLR_STRUCT(composite_mode);
			composite_mode.opacity = 255;
			composite_mode.xfer = GetXferMode(infoP->xfer);

			for (LayerInfo& layer : infoP->pack) {
				ERR((extra->cb->checkout_layer_pixels(in_data->effect_ref, layer.idL, &middle_worldP)));

				ERR(in_data->utils->transform_world(
					in_data->effect_ref,
					in_data->quality,
					in_data->in_flags,
					in_data->field,
					middle_worldP,
					&composite_mode,
					NULL,
					&layer.matrix,
					1,
					true,
					&middle_worldP->extent_hint,
					output_worldP)
				);
			}
		} else {
			ERR(PF_COPY(input_worldP, output_worldP, NULL, NULL));
		}
		suites.HandleSuite1()->host_unlock_handle(reinterpret_cast<PF_Handle>(extra->input->pre_render_data));
	} else {
		err = PF_Err_BAD_CALLBACK_PARAM;
	}
	//ERR2(AEFX_ReleaseSuite(in_data,
	//	out_data,
	//	kPFWorldSuite,
	//	kPFWorldSuiteVersion2,
	//	"Couldn't release suite."));
	return err;
}

extern "C" DllExport
PF_Err PluginDataEntryFunction2(
	PF_PluginDataPtr inPtr,
	PF_PluginDataCB2 inPluginDataCallBackPtr,
	SPBasicSuite* inSPBasicSuitePtr,
	const char* inHostName,
	const char* inHostVersion)
{
	PF_Err result = PF_Err_INVALID_CALLBACK;

	result = PF_REGISTER_EFFECT_EXT2(
		inPtr,
		inPluginDataCallBackPtr,
		PLUGIN_ID, // Name
		PLUGIN_ID, // Match Name
		"Otomad Useful Plug-ins", // Category
		AE_RESERVED_INFO, // Reserved Info
		"EffectMain",	// Entry point
		"https://otm.ink/");	// support URL

	return result;
}

PF_Err
EffectMain(
	PF_Cmd			cmd,
	PF_InData		*in_data,
	PF_OutData		*out_data,
	PF_ParamDef		*params[],
	PF_LayerDef		*output,
	void			*extra)
{
	PF_Err		err = PF_Err_NONE;
	
	try {
		switch (cmd) {
			case PF_Cmd_ABOUT:

				err = About(in_data,
							out_data,
							params,
							output);
				break;
				
			case PF_Cmd_GLOBAL_SETUP:

				err = GlobalSetup(	in_data,
									out_data,
									params,
									output);
				break;
				
			case PF_Cmd_PARAMS_SETUP:

				err = ParamsSetup(	in_data,
									out_data,
									params,
									output);
				break;

			case PF_Cmd_RENDER:

				err = Render(	in_data,
								out_data,
								params,
								output);
				break;

			case PF_Cmd_SMART_PRE_RENDER:
				err = PreRender(in_data, out_data, (PF_PreRenderExtra*)extra);
				break;

			case PF_Cmd_SMART_RENDER:
				err = SmartRender(in_data, out_data, (PF_SmartRenderExtra*)extra);
				break;
		}
	}
	catch(PF_Err &thrown_err){
		err = thrown_err;
	}
	return err;
}