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

/*
	Skeleton.h
*/

#pragma once

#ifndef SKELETON_H
#define SKELETON_H

typedef unsigned char		u_char;
typedef unsigned short		u_short;
typedef unsigned short		u_int16;
typedef unsigned long		u_long;
typedef short int			int16;
#define PF_TABLE_BITS	12
#define PF_TABLE_SZ_16	4096

#define PF_DEEP_COLOR_AWARE 1	// make sure we get 16bpc pixels; 
								// AE_Effect.h checks for this.

#include "AEConfig.h"

#ifdef AE_OS_WIN
	typedef unsigned short PixelType;
	#include <Windows.h>
#endif

#include "entry.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_Macros.h"
#include "Param_Utils.h"
#include "AE_EffectCBSuites.h"
#include "String_Utils.h"
#include "AE_GeneralPlug.h"
#include "AEFX_ChannelDepthTpl.h"
#include "AEGP_SuiteHandler.h"

#include "KTrigger_Strings.h"
#include <vector>

#define PLUGIN_ID "Keyframe Trigger"

/* Versioning information */

#define	MAJOR_VERSION	2
#define	MINOR_VERSION	1
#define	BUG_VERSION		0
#define	STAGE_VERSION	PF_Stage_RELEASE
#define	BUILD_VERSION	1

/* RESOURCE_VERSION =
MAJOR_VERSION * 524288 +
MINOR_VERSION * 32768 +
BUG_VERSION * 2048
STAGE_VERSION * 512
BUILD_VERSION */

/* Parameter defaults */

#define	SKELETON_GAIN_MIN		-10000
#define	SKELETON_GAIN_MAX		10000
#define	SKELETON_GAIN_DFLT		0

enum {
	SKELETON_INPUT = 0,
	SKELETON_GAIN,
	START_TIME_DISK,
	PLAYRATE_DISK,
	IS_CONTROL_DISK,
	IS_FROZEN_DISK,
	ANCHOR_DISK,
	SCALE_SIZE_DISK,
	IS_FLIP_DISK,
	SWITCH_DISK,
	OFFSET_COUNT_DISK,
	RETURN_GROUP_DISK,
	RETURN_DISK,
	RETURN2_DISK,
	RETURN_GROUP_END_DISK,
	OFFSET_GROUP,
	OFFSET_DISK,
	OFFSET_GROUP_END,
	ROTATE_GROUP,
	ROTATE_DISK,
	ROTATE_SCALE_DISK,
	ROTATE_GROUP_END,
	SKELETON_MAX_DUR,
	XFER_MODE_DISK,
	SECOND_LAYER_DISK,
	RANDOM_DISK,
	SKELETON_NUM_PARAMS
};

enum {
	GAIN_DISK_ID = 1,
	START_TIME_DISK_ID,
	PLAYRATE_DISK_ID,
	IS_CONTROL_DISK_ID,
	IS_FROZEN_DISK_ID,
	ANCHOR_DISK_ID,
	SCALE_SIZE_DISK_ID,
	IS_FLIP_DISK_ID,
	SWITCH_DISK_ID,
	OFFSET_COUNT_DISK_ID,
	RETURN_GROUP_DISK_ID,
	RETURN_DISK_ID,
	RETURN2_DISK_ID,
	RETURN_GROUP_END_DISK_ID,
	OFFSET_GROUP_ID,
	OFFSET_DISK_ID,
	OFFSET_GROUP_END_ID,
	ROTATE_GROUP_ID,
	ROTATE_DISK_ID,
	ROTATE_SCALE_DISK_ID,
	ROTATE_GROUP_END_ID,
	MAX_DUR_DISK_ID,
	XFER_MODE_DISK_ID,
	SECOND_LAYER_DISK_ID,
	RANDOM_DISK_ID
};

typedef struct GainInfo{
	PF_FpLong	gainF;
} GainInfo, *GainInfoP, **GainInfoH;

typedef struct LayerInfo {
	A_long idL;
	PF_FloatMatrix matrix;
};

typedef struct LayerPack {
	A_long     xfer;
	std::vector<LayerInfo> pack;
} LayerPack;

extern "C" {

	DllExport
	PF_Err
	EffectMain(
		PF_Cmd			cmd,
		PF_InData		*in_data,
		PF_OutData		*out_data,
		PF_ParamDef		*params[],
		PF_LayerDef		*output,
		void			*extra);

}

#endif // SKELETON_H