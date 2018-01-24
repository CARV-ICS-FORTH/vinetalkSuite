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

import com.sun.jna.Native;
import com.sun.jna.NativeLibrary;
import com.sun.jna.Pointer;
import com.sun.jna.Memory;
import com.sun.jna.ptr.PointerByReference;
import java.io.Serializable;
import java.util.*;

public class Vinetalk implements Serializable
{
	public Vinetalk()
	{
		init();
	}
	
	public void init()
	{
		NativeLibrary.getInstance("rt");
		VineTalkInterface.INSTANCE.vine_talk_init();
	}

	public static VineProcedure acquireProcedure(VineAccelerator.Type type,String name)
	{
		Pointer proc;

		proc = VineTalkInterface.INSTANCE.vine_proc_get(type.getAsInt(),name);

		if(proc == Pointer.NULL)
			return null;
			
		return new VineProcedure(proc);
	}

	public static  VineAccelerator[] listAccelerators(VineAccelerator.Type type,Boolean physical)
	{
		PointerByReference ptr_ref = new PointerByReference();
		int accels = VineTalkInterface.INSTANCE.vine_accel_list(type.getAsInt(),physical,ptr_ref);
		System.out.println("Found "+accels+" accelerators");
		VineAccelerator [] accel_ar = new VineAccelerator[accels];
		int i = 0;
		for( Pointer ptr : ptr_ref.getValue().getPointerArray(0,accels) )
		{
			accel_ar[i++] = new VineAccelerator(ptr);
		}
		// Free ptr_ref
		Native.free(Pointer.nativeValue(ptr_ref.getValue())); // Not sure if this actually works...
		return accel_ar;
	}

	public static VineAccelerator acquireAccelerator(VineAccelerator.Type type)
	{
		return new VineAccelerator(VineTalkInterface.INSTANCE.vine_accel_acquire_type(type.getAsInt()));
	}

	public static VineAccelerator acquireAccelerator (VineAccelerator accel)
	{
		PointerByReference ptr_ref = new PointerByReference(accel.getPointer());
		VineTalkInterface.INSTANCE.vine_accel_acquire_phys(ptr_ref);
		return new VineAccelerator(ptr_ref.getValue());
	}

	public void exit()
	{
		VineTalkInterface.INSTANCE.vine_talk_exit();
	}
}
