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
#include "system.h"
#include <sys/stat.h>
#include <unistd.h>
#include <pwd.h>

size_t system_total_memory()
{
	size_t pages     = sysconf(_SC_PHYS_PAGES);
	size_t page_size = sysconf(_SC_PAGE_SIZE);

	return pages * page_size;
}

char* system_home_path()
{
	uid_t         uid = getuid();
	struct passwd *pw = getpwuid(uid);

	if (!pw)
		return 0;

	return pw->pw_dir;
}

int system_compare_ptrs(const void * a,const void * b)
{
	return (int)((size_t)a - (size_t)b);
}

off_t system_file_size(const char * file)
{
	struct stat stats = {0};
	if(stat(file,&stats))
		return 0;
	return stats.st_size;
}
