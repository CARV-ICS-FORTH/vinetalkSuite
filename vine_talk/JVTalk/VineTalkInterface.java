/*
 * Copyright 2018 Foundation for Research and Technology - Hellas
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at 
 * http://www.apache.org/licenses/LICENSE-2.0 [1] [1] 
 * 
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
 * See the License for the specific language governing permissions and 
 * limitations under the License. 
 * 
 * Links: 
 * 
 * [1] http://www.apache.org/licenses/LICENSE-2.0 [1] 
 */
package Vinetalk;
import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Platform;
import com.sun.jna.ptr.PointerByReference;
import com.sun.jna.Pointer;
import com.sun.jna.Structure;


public interface VineTalkInterface extends Library
{
	VineTalkInterface INSTANCE = (VineTalkInterface)Native.loadLibrary("vine",VineTalkInterface.class);

	void vine_talk_init();
	int vine_accel_list(int type, boolean physical, PointerByReference accels);
	Pointer vine_accel_acquire_type (int type);
	int vine_accel_acquire_phys(PointerByReference accel);
	int vine_accel_release(PointerByReference accel);
	int vine_vaccel_queue_size(Pointer vaccel);
	Pointer vine_proc_get(int type,String func_name);
	Pointer vine_task_issue (Pointer accel, Pointer proc, Pointer args, long in_count, Structure[] input, long out_count, Structure[] output);
	int vine_task_wait (Pointer task);
	int vine_task_stat (Pointer task,Pointer stat);
	int vine_task_free (Pointer task);
	int vine_proc_put(Pointer proc);
	void vine_talk_exit();
}
