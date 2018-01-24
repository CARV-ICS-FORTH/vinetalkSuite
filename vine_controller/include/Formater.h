#ifndef FORMATER_HEADER
#define FORMATER_HEADER
#include <ostream>
#include <vector>

template <class CLASS>
class Formater
{
    public:
        template< class ITEM >
            friend ostream & operator<<(ostream & os,Formater< ITEM > fmt);
        Formater(const char * pre,const char * aft,const char * fin,const vector< CLASS > & vec)
            : pre(pre), aft(aft), fin(fin), vec(vec) {}
    private:
        const char * pre;				// Before output
        const char * aft;				// Between items
        const char * fin;				// After output
        const vector< CLASS > & vec;	// Stuff to output
};

template<class CLASS>
ostream & operator<<(ostream & os,Formater< CLASS > fmt)
{
    const char * delim = "";
    os << fmt.pre;
    for(typename vector< CLASS >::const_iterator itr = fmt.vec.begin() ;
            itr != fmt.vec.end() ; itr++)
    {
        os << delim << *itr;
        delim = fmt.aft;
    }
    os << fmt.fin;
    return os;
}
#endif
