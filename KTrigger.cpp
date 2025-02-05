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

	out_data->out_flags = PF_OutFlag_NON_PARAM_VARY;
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
							SKELETON_GAIN_MIN,
							SKELETON_GAIN_MAX,
							SKELETON_GAIN_DFLT,
							PF_Precision_HUNDREDTHS,
							0,
							0,
							GAIN_DISK_ID);

							AEFX_CLR_STRUCT(def);

	// 控制器影响
	PF_ADD_CHECKBOXX(STR(StrID_Is_Control_Param_Name),
		false,
		0,
		IS_CONTROL_DISK_ID);

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
		10000,
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

	(*transform_matrix).mat[2][0] = width / 2;
	(*transform_matrix).mat[2][1] = height / 2;
}

static void Flip2Matrix(A_long width, A_long height, PF_FloatMatrix* transform_matrix) {

	(*transform_matrix).mat[0][0] = -1;

	(*transform_matrix).mat[1][1] = 1;

	(*transform_matrix).mat[2][2] = 1;

	(*transform_matrix).mat[2][0] = width;
	//(*transform_matrix).mat[2][1] = height;
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

	/*	Put interesting code here. */
	GainInfo			giP;
	AEFX_CLR_STRUCT(giP);

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
	// 从第一帧开始遍历，如果在时间范围内就叠加上去
	for (int i = 0;i < keyframes;i++) {
		A_long key_time = 0;
		A_u_long key_time_scale = 0;
		ERR(suites.ParamUtilsSuite3()->PF_KeyIndexToTime(
			in_data->effect_ref,
			SKELETON_GAIN,
			i,
			&key_time,
			&key_time_scale));
		// 如果在时间范围外，则忽略
		if (ctime < key_time ||
			ctime - key_time > in_data->total_time){
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
		// 获取相应时间画面
		PF_ParamDef imageDef;
		AEFX_CLR_STRUCT(imageDef);
		ERR(PF_CHECKOUT_PARAM(
			in_data,
			PF_Param_LAYER,
			ctime - key_time,
			in_data->time_step,
			in_data->time_scale,
			&imageDef));
		// 获取当前帧数据（如果启用，否则为索引）
		PF_FpLong state = 0;
		const A_long offset_count = params[OFFSET_COUNT_DISK_ID]->u.fs_d.value;
		if (params[IS_CONTROL_DISK_ID]->u.bd.value) {
			PF_ParamDef* ckey;
			AEFX_CLR_STRUCT(ckey);
			ERR(suites.ParamUtilsSuite3()->PF_CheckoutKeyframe(in_data->effect_ref,
				SKELETON_GAIN,
				i,
				NULL,
				NULL,
				ckey));
			state = ckey ? ckey->u.fs_d.value : 0;
			ERR(suites.ParamUtilsSuite3()->PF_CheckinKeyframe(
				in_data->effect_ref,
				ckey));
		}
		else {
			if (offset_count > 0) {
				if (params[RETURN_DISK_ID]->u.bd.value) {
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
		// 水平翻转
		if (params[IS_FLIP_DISK_ID]->u.bd.value && i % 2 == 1) {
			PF_EffectWorld cworld;
			AEFX_CLR_STRUCT(cworld);
			ERR(PF_NEW_WORLD(
				imageDef.u.ld.width,
				imageDef.u.ld.height,
				NULL,
				&cworld));
			PF_FloatMatrix flip_matrix;
			Flip2Matrix(
				in_data->extent_hint.right - in_data->extent_hint.left,
				in_data->extent_hint.bottom - in_data->extent_hint.top,
				&flip_matrix);
			ERR(in_data->utils->transform_world(
				in_data->effect_ref,
				in_data->quality,
				in_data->in_flags,
				in_data->field,
				&imageDef.u.ld,
				&composite_mode,
				NULL,
				&flip_matrix,
				1,
				true,
				&in_data->extent_hint,
				&cworld)
			);
			imageDef.u.ld = cworld;
			//PF_COPY(&cworld, &imageDef.u.ld, NULL, NULL);
			//PF_DISPOSE_WORLD(&cworld);
		}
		// 混合进新画面中
		switch (params[SWITCH_DISK_ID]->u.pd.value) {
		case 1: {
			PF_FpLong x_offset = FIX_2_FLOAT(params[OFFSET_DISK_ID]->u.td.x_value);
			PF_FpLong y_offset = FIX_2_FLOAT(params[OFFSET_DISK_ID]->u.td.y_value);

			ERR(in_data->utils->transfer_rect(
				in_data->effect_ref,
				in_data->quality,
				in_data->in_flags,
				in_data->field,
				&in_data->extent_hint,
				&imageDef.u.ld,
				&composite_mode,
				NULL,
				state * x_offset,
				state * y_offset,
				output)
			);
			break;
		}
		case 2: {
			const PF_FpLong angle = FIX_2_FLOAT(params[ROTATE_DISK_ID]->u.ad.value) * state;
			PF_FloatMatrix matrix;
			Angle2Matrix(angle, 
				in_data->extent_hint.right - in_data->extent_hint.left,
				in_data->extent_hint.bottom - in_data->extent_hint.top,
				params[ROTATE_SCALE_DISK_ID]->u.fs_d.value / 100,
				&matrix);
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
					output)
			);
			break;
			}
		default: {
			break;
		}
		}
		ERR(PF_CHECKIN_PARAM(in_data, &imageDef));
	}

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
		}
	}
	catch(PF_Err &thrown_err){
		err = thrown_err;
	}
	return err;
}