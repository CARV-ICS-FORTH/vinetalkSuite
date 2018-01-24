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

import java.util.ArrayList;
import com.sun.jna.Pointer;
import com.sun.jna.Structure;
import java.io.Serializable;

public class VineTask implements Serializable
{
    public enum State {
        Failed(0), /**< Task execution failed. */
        Issued(1), /**< Task has been issued. */
        Completed(2); /**< Task has been completed. */
        private final int value;

        State(int value)
        {this.value = value;}
        int getAsInt()
        {return value;}
    }

	public VineTask(VineProcedure proc,Structure args)
	{
		this.proc = proc;
		this.args = args;
		this.inputs = new ArrayList<VineBuffer>();
		this.outputs = new ArrayList<VineBuffer>();
	}

	public VineTask(VineProcedure proc)
	{
		this.proc = proc;
		this.inputs = new ArrayList<VineBuffer>();
		this.outputs = new ArrayList<VineBuffer>();
	}

	public Pointer getProcedure()
	{
		if( proc != null)
			return proc.getPointer();
		else
			return null;
	}

	public VineBuffer getArgs()
	{
		if(args == null)
			return null;
		args.write();
		return new VineBuffer(args);
	}

	public VineBuffer[] getInputs()
	{
		if(inputs.size() > 0)
		{
			VineBuffer temp = inputs.get(0);
			VineBuffer [] ins = (VineBuffer [])temp.toArray(inputs.size());
			ArrayList<VineBuffer> new_inputs = new ArrayList<VineBuffer>();
			for(int c = 0 ; c < inputs.size() ; c++)
			{
				ins[c].copyFrom(inputs.get(c));
				new_inputs.add(ins[c]);
			}
			inputs = new_inputs;
			return ins;
		}
		return null;
	}

	public VineBuffer[] getOutputs()
	{
		if(outputs.size() > 0)
		{
			VineBuffer temp = outputs.get(0);
			VineBuffer [] outs = (VineBuffer [])temp.toArray(outputs.size());
			ArrayList<VineBuffer> new_outputs = new ArrayList<VineBuffer>();
			for(int c = 0 ; c < outputs.size() ; c++)
			{
				outs[c].copyFrom(outputs.get(c));
				new_outputs.add(outs[c]);
			}
			outputs = new_outputs;
			return outs;
		}
		return null;
	}

	public void setArgs(Structure args)
	{
		this.args = args;
		args.write();
	}

	public void setTask(Pointer task)
	{
		this.task = task;
	}

	public void addInput(byte [] data)
	{
		inputs.add(new VineBuffer(data));
	}

	public void addInput(float [] data)
	{
		inputs.add(new VineBuffer(data));
	}

	public void addInput(long [] data)
	{
		inputs.add(new VineBuffer(data));
	}

	public void addInput(int [] data)
	{
		inputs.add(new VineBuffer(data));
	}
	
	public void addInput(Structure data)
	{
		inputs.add(new VineBuffer(data));
	}
	
	public void addInput(Structure data, int elements)
	{
		inputs.add(new VineBuffer(data, elements));
	}
	
	public void addOutput(byte [] data)
	{
		outputs.add(new VineBuffer(data,false));
	}

	public void addOutput(float [] data)
	{
		outputs.add(new VineBuffer(data,false));
	}

	public void addOutput(long [] data)
	{
		outputs.add(new VineBuffer(data,false));
	}

	public void addOutput(int [] data)
	{
		outputs.add(new VineBuffer(data,false));
	}
	
	public void addOutput(Structure data)
	{
		outputs.add(new VineBuffer(data,false));
	}
	
    public void addOutput(Structure data, int elements)
	{
		outputs.add(new VineBuffer(data, elements, false));
	}
	
	public State status()
	{
		return status(true);
	}

	public State status(boolean sync)
	{
		int ret;

		if(sync) 
			ret = VineTalkInterface.INSTANCE.vine_task_wait(task);
		else
			ret = VineTalkInterface.INSTANCE.vine_task_stat(task,null);
        
		if(ret == 2) // Complete
		{
			if(!sync)       // Call wait since it will not block
				VineTalkInterface.INSTANCE.vine_task_wait(task);

			for(VineBuffer vb : outputs) {
				vb.read();
			}

			VineTalkInterface.INSTANCE.vine_task_free(task);
		}
		return State.values()[ret];
	}

	private VineProcedure proc;
	private Structure args;
	private Pointer task;
	public ArrayList<VineBuffer> inputs;
	public ArrayList<VineBuffer> outputs;
}
