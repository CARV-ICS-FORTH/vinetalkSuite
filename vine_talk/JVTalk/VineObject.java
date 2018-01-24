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
import com.sun.jna.Pointer;
import com.sun.jna.Structure;
import java.util.List;
import java.util.Arrays;

public abstract class VineObject
{
	public class cRep extends Structure
	{
		public Pointer prev;
		public Pointer next;
		public Pointer owner;
		/* TODO:Probably should make the above a seperate struct */
		public int type;
		public byte[] name = new byte[32];	// Not sure if 'proper'
		public cRep(Pointer ptr)
		{
			super(ptr);
			read();
		}
		protected List<String> getFieldOrder()
		{
			return Arrays.asList(new String[] { "prev", "next", "owner","type","name"});
		}
	}

	public VineObject(Pointer ptr)
	{
		this.ptr = ptr;
		crep = new cRep(ptr);
	}

	public String getName()
	{
		return new String(crep.name);
	}

	public Pointer getPointer()
	{
		return ptr;
	}

	public void setPointer(Pointer ptr)
	{
		this.ptr = ptr;
	}

	public String toString()
	{
		return this.getClass().getName()+"("+new String(crep.name)+","+crep.type+")@0x"+Long.toHexString(Pointer.nativeValue(crep.owner));
	}
	public abstract void release();
	private Pointer ptr;
	private cRep crep;
}
