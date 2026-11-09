#include "level.hpp"
int main(){return clamp_level(-1)!=0 || clamp_level(50)!=50 || clamp_level(100)!=100;}
