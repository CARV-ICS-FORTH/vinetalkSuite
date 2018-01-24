#include "Utilities.h"
#include <sstream>
#include <algorithm>

std::map<std::string,std::string> decodeArgs(std::string args)
{
	std::string k,v;
	std::istringstream iss(args);
	std::map<std::string,std::string> kv;

	do
	{
		std::getline(iss,k,':');
		std::getline(iss,v,',');
		if(iss)
		{
			k = trim(k);
			std::transform(k.begin(), k.end(), k.begin(), ::tolower);
			v = trim(v);
			kv[k] = v;
		}
	}while(iss);

	return kv;
}

std::string trim(const std::string &s)
{
	auto wsfront=std::find_if_not(s.begin(),s.end(),[](int c){return std::isspace(c);});
	auto wsback=std::find_if_not(s.rbegin(),s.rend(),[](int c){return std::isspace(c);}).base();
	return (wsback<=wsfront ? std::string() : std::string(wsfront,wsback));
}
