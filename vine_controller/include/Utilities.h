#ifndef VINE_CONTROLLER_UTILITIES_HEADER_FILE
	#define VINE_CONTROLLER_UTILITIES_HEADER_FILE
	#include <map>
	#include <string>

	/**
	 * Convert arguement list of the following format to a map:
	 *
	 * key1:val1,key2:val2
	 *
	 * Resulting map has keys and values in lowercase.
	 * Keys and values are assumed to be words
	 * (i.e a key/value can not contain whitespace)
	 */
	std::map<std::string,std::string> decodeArgs(std::string args);

	/*
	 * Trim whitespace from word contained in \c word.
	 *
	 * Removes whitespace from the begining and end of \c word.
	 */
	std::string trim(const std::string &s);
#endif
