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
#include "list.h"

utils_list_s* utils_list_init(void *mem)
{
	utils_list_s *list = mem;

	list->length = 0;
	utils_list_node_init( &(list->head) , list );
	return list;
}

void utils_list_node_add(utils_list_node_s *head, utils_list_node_s *node)
{
	head->next->prev = node;
	node->next       = head->next;
	node->prev       = head;
	head->next       = node;
}

void utils_list_add(utils_list_s *list, utils_list_node_s *node)
{
	utils_list_node_add(&(list->head), node);
	list->length++;
}

utils_list_node_s* utils_list_del(utils_list_s *list, utils_list_node_s *node)
{
	node->next->prev = node->prev;
	node->prev->next = node->next;
	list->length--;
	return node;
}

size_t utils_list_to_array(utils_list_s *list, utils_list_node_s **array)
{
	utils_list_node_s *itr;

	if (!array)
		return list->length;

	if (list->length)
		utils_list_for_each(*list, itr) {
			*array = itr->owner;
			array++;
		}
	return list->length;
}

void utils_list_node_init(utils_list_node_s *node,void * owner)
{
	node->next = node;
	node->prev = node;
	node->owner = owner;
}
