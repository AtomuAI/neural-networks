// Copyright 2024 Shane W. Mulcahy

#ifndef BEWUSSTSEIN_NEURAL_NETWORKS_CPP_MACRO_FOR_HPP_
#define BEWUSSTSEIN_NEURAL_NETWORKS_CPP_MACRO_FOR_HPP_

#define FOR_1( macro, a )\
    macro( a )
#define FOR_2( macro, a, b )\
    FOR_1( macro, a )\
    FOR_1( macro, b )
#define FOR_3( macro, a, b, c )\
    FOR_2( macro, a, b )\
    FOR_1( macro, c )
#define FOR_4( macro, a, b, c, d )\
    FOR_3( macro, a, b, c )\
    FOR_1( macro, d )
#define FOR_5( macro, a, b, c, d, e )\
    FOR_4( macro, a, b, c, d )\
    FOR_1( macro, e )
#define FOR_6( macro, a, b, c, d, e, f )\
    FOR_5( macro, a, b, c, d, e )\
    FOR_1( macro, f )
#define FOR_7( macro, a, b, c, d, e, f, g )\
    FOR_6( macro, a, b, c, d, e, f )\
    FOR_1( macro, g )
#define FOR_8( macro, a, b, c, d, e, f, g, h )\
    FOR_7( macro, a, b, c, d, e, f, g )\
    FOR_1( macro, h )
#define FOR_9( macro, a, b, c, d, e, f, g, h, i )\
    FOR_8( macro, a, b, c, d, e, f, g, h )\
    FOR_1( macro, i )
#define FOR_10( macro, a, b, c, d, e, f, g, h, i, j )\
    FOR_9( macro, a, b, c, d, e, f, g, h, i )\
    FOR_1( macro, j )

#define FOR_1_FOR_1( macro, a1, b1 )\
    macro( a1, b1 )
#define FOR_1_FOR_2( macro, a1, b1, b2 )\
    FOR_1_FOR_1( macro, a1, b1 )\
    FOR_1_FOR_1( macro, a1, b2 )
#define FOR_1_FOR_3( macro, a1, b1, b2, b3 )\
    FOR_1_FOR_2( macro, a1, b1, b2 )\
    FOR_1_FOR_1( macro, a1, b3 )
#define FOR_1_FOR_4( macro, a1, b1, b2, b3, b4 )\
    FOR_1_FOR_3( macro, a1, b1, b2, b3 )\
    FOR_1_FOR_1( macro, a1, b4 )
#define FOR_1_FOR_5( macro, a1, b1, b2, b3, b4, b5 )\
    FOR_1_FOR_4( macro, a1, b1, b2, b3, b4 )\
    FOR_1_FOR_1( macro, a1, b5 )
#define FOR_1_FOR_6( macro, a1, b1, b2, b3, b4, b5, b6 )\
    FOR_1_FOR_5( macro, a1, b1, b2, b3, b4, b5 )\
    FOR_1_FOR_1( macro, a1, b6 )
#define FOR_1_FOR_7( macro, a1, b1, b2, b3, b4, b5, b6, b7 )\
    FOR_1_FOR_6( macro, a1, b1, b2, b3, b4, b5, b6 )\
    FOR_1_FOR_1( macro, a1, b7 )
#define FOR_1_FOR_8( macro, a1, b1, b2, b3, b4, b5, b6, b7, b8 )\
    FOR_1_FOR_7( macro, a1, b1, b2, b3, b4, b5, b6, b7 )\
    FOR_1_FOR_1( macro, a1, b8 )
#define FOR_1_FOR_9( macro, a1, b1, b2, b3, b4, b5, b6, b7, b8, b9 )\
    FOR_1_FOR_8( macro, a1, b1, b2, b3, b4, b5, b6, b7, b8 )\
    FOR_1_FOR_1( macro, a1, b9 )
#define FOR_1_FOR_10( macro, a1, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10 )\
    FOR_1_FOR_9( macro, a1, b1, b2, b3, b4, b5, b6, b7, b8, b9 )\
    FOR_1_FOR_1( macro, a1, b10 )

#define FOR_2_FOR_1( macro, a1, a2, b1 )\
    FOR_1_FOR_1( macro, a1, b1 )\
    FOR_1_FOR_1( macro, a2, b1 )
#define FOR_2_FOR_2( macro, a1, a2, b1, b2 )\
    FOR_2_FOR_1( macro, a1, a2, b1 )\
    FOR_2_FOR_1( macro, a1, a2, b2 )
#define FOR_2_FOR_3( macro, a1, a2, b1, b2, b3 )\
    FOR_2_FOR_2( macro, a1, a2, b1, b2 )\
    FOR_2_FOR_1( macro, a1, a2, b3 )
#define FOR_2_FOR_4( macro, a1, a2, b1, b2, b3, b4 )\
    FOR_2_FOR_3( macro, a1, a2, b1, b2, b3 )\
    FOR_2_FOR_1( macro, a1, a2, b4 )
#define FOR_2_FOR_5( macro, a1, a2, b1, b2, b3, b4, b5 )\
    FOR_2_FOR_4( macro, a1, a2, b1, b2, b3, b4 )\
    FOR_2_FOR_1( macro, a1, a2, b5 )
#define FOR_2_FOR_6( macro, a1, a2, b1, b2, b3, b4, b5, b6 )\
    FOR_2_FOR_5( macro, a1, a2, b1, b2, b3, b4, b5 )\
    FOR_2_FOR_1( macro, a1, a2, b6 )
#define FOR_2_FOR_7( macro, a1, a2, b1, b2, b3, b4, b5, b6, b7 )\
    FOR_2_FOR_6( macro, a1, a2, b1, b2, b3, b4, b5, b6 )\
    FOR_2_FOR_1( macro, a1, a2, b7 )
#define FOR_2_FOR_8( macro, a1, a2, b1, b2, b3, b4, b5, b6, b7, b8 )\
    FOR_2_FOR_7( macro, a1, a2, b1, b2, b3, b4, b5, b6, b7 )\
    FOR_2_FOR_1( macro, a1, a2, b8 )
#define FOR_2_FOR_9( macro, a1, a2, b1, b2, b3, b4, b5, b6, b7, b8, b9 )\
    FOR_2_FOR_8( macro, a1, a2, b1, b2, b3, b4, b5, b6, b7, b8 )\
    FOR_2_FOR_1( macro, a1, a2, b9 )
#define FOR_2_FOR_10( macro, a1, a2, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10 )\
    FOR_2_FOR_9( macro, a1, a2, b1, b2, b3, b4, b5, b6, b7, b8, b9 )\
    FOR_2_FOR_1( macro, a1, a2, b10 )

#define FOR_3_FOR_1( macro, a1, a2, a3, b1 )\
    FOR_2_FOR_1( macro, a1, a2, b1 )\
    FOR_1_FOR_1( macro, a3, b1 )
#define FOR_3_FOR_2( macro, a1, a2, a3, b1, b2 )\
    FOR_3_FOR_1( macro, a1, a2, a3, b1 )\
    FOR_3_FOR_1( macro, a1, a2, a3, b2 )
#define FOR_3_FOR_3( macro, a1, a2, a3, b1, b2, b3 )\
    FOR_3_FOR_2( macro, a1, a2, a3, b1, b2 )\
    FOR_3_FOR_1( macro, a1, a2, a3, b3 )
#define FOR_3_FOR_4( macro, a1, a2, a3, b1, b2, b3, b4 )\
    FOR_3_FOR_3( macro, a1, a2, a3, b1, b2, b3 )\
    FOR_3_FOR_1( macro, a1, a2, a3, b4 )
#define FOR_3_FOR_5( macro, a1, a2, a3, b1, b2, b3, b4, b5 )\
    FOR_3_FOR_4( macro, a1, a2, a3, b1, b2, b3, b4 )\
    FOR_3_FOR_1( macro, a1, a2, a3, b5 )
#define FOR_3_FOR_6( macro, a1, a2, a3, b1, b2, b3, b4, b5, b6 )\
    FOR_3_FOR_5( macro, a1, a2, a3, b1, b2, b3, b4, b5 )\
    FOR_3_FOR_1( macro, a1, a2, a3, b6 )
#define FOR_3_FOR_7( macro, a1, a2, a3, b1, b2, b3, b4, b5, b6, b7 )\
    FOR_3_FOR_6( macro, a1, a2, a3, b1, b2, b3, b4, b5, b6 )\
    FOR_3_FOR_1( macro, a1, a2, a3, b7 )
#define FOR_3_FOR_8( macro, a1, a2, a3, b1, b2, b3, b4, b5, b6, b7, b8 )\
    FOR_3_FOR_7( macro, a1, a2, a3, b1, b2, b3, b4, b5, b6, b7 )\
    FOR_3_FOR_1( macro, a1, a2, a3, b8 )
#define FOR_3_FOR_9( macro, a1, a2, a3, b1, b2, b3, b4, b5, b6, b7, b8, b9 )\
    FOR_3_FOR_8( macro, a1, a2, a3, b1, b2, b3, b4, b5, b6, b7, b8 )\
    FOR_3_FOR_1( macro, a1, a2, a3, b9 )
#define FOR_3_FOR_10( macro, a1, a2, a3, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10 )\
    FOR_3_FOR_9( macro, a1, a2, a3, b1, b2, b3, b4, b5, b6, b7, b8, b9 )\
    FOR_3_FOR_1( macro, a1, a2, a3, b10 )

#define FOR_4_FOR_1( macro, a1, a2, a3, a4, b1 )\
    FOR_3_FOR_1( macro, a1, a2, a3, b1 )\
    FOR_1_FOR_1( macro, a4, b1 )
#define FOR_4_FOR_2( macro, a1, a2, a3, a4, b1, b2 )\
    FOR_4_FOR_1( macro, a1, a2, a3, a4, b1 )\
    FOR_4_FOR_1( macro, a1, a2, a3, a4, b2 )
#define FOR_4_FOR_3( macro, a1, a2, a3, a4, b1, b2, b3 )\
    FOR_4_FOR_2( macro, a1, a2, a3, a4, b1, b2 )\
    FOR_4_FOR_1( macro, a1, a2, a3, a4, b3 )
#define FOR_4_FOR_4( macro, a1, a2, a3, a4, b1, b2, b3, b4 )\
    FOR_4_FOR_3( macro, a1, a2, a3, a4, b1, b2, b3 )\
    FOR_4_FOR_1( macro, a1, a2, a3, a4, b4 )
#define FOR_4_FOR_5( macro, a1, a2, a3, a4, b1, b2, b3, b4, b5 )\
    FOR_4_FOR_4( macro, a1, a2, a3, a4, b1, b2, b3, b4 )\
    FOR_4_FOR_1( macro, a1, a2, a3, a4, b5 )
#define FOR_4_FOR_6( macro, a1, a2, a3, a4, b1, b2, b3, b4, b5, b6 )\
    FOR_4_FOR_5( macro, a1, a2, a3, a4, b1, b2, b3, b4, b5 )\
    FOR_4_FOR_1( macro, a1, a2, a3, a4, b6 )
#define FOR_4_FOR_7( macro, a1, a2, a3, a4, b1, b2, b3, b4, b5, b6, b7 )\
    FOR_4_FOR_6( macro, a1, a2, a3, a4, b1, b2, b3, b4, b5, b6 )\
    FOR_4_FOR_1( macro, a1, a2, a3, a4, b7 )
#define FOR_4_FOR_8( macro, a1, a2, a3, a4, b1, b2, b3, b4, b5, b6, b7, b8 )\
    FOR_4_FOR_7( macro, a1, a2, a3, a4, b1, b2, b3, b4, b5, b6, b7 )\
    FOR_4_FOR_1( macro, a1, a2, a3, a4, b8 )
#define FOR_4_FOR_9( macro, a1, a2, a3, a4, b1, b2, b3, b4, b5, b6, b7, b8, b9 )\
    FOR_4_FOR_8( macro, a1, a2, a3, a4, b1, b2, b3, b4, b5, b6, b7, b8 )\
    FOR_4_FOR_1( macro, a1, a2, a3, a4, b9 )
#define FOR_4_FOR_10( macro, a1, a2, a3, a4, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10 )\
    FOR_4_FOR_9( macro, a1, a2, a3, a4, b1, b2, b3, b4, b5, b6, b7, b8, b9 )\
    FOR_4_FOR_1( macro, a1, a2, a3, a4, b10 )

#define FOR_5_FOR_1( macro, a1, a2, a3, a4, a5, b1 )\
    FOR_4_FOR_1( macro, a1, a2, a3, a4, b1 )\
    FOR_1_FOR_1( macro, a5, b1 )
#define FOR_5_FOR_2( macro, a1, a2, a3, a4, a5, b1, b2 )\
    FOR_5_FOR_1( macro, a1, a2, a3, a4, a5, b1 )\
    FOR_5_FOR_1( macro, a1, a2, a3, a4, a5, b2 )
#define FOR_5_FOR_3( macro, a1, a2, a3, a4, a5, b1, b2, b3 )\
    FOR_5_FOR_2( macro, a1, a2, a3, a4, a5, b1, b2 )\
    FOR_5_FOR_1( macro, a1, a2, a3, a4, a5, b3 )
#define FOR_5_FOR_4( macro, a1, a2, a3, a4, a5, b1, b2, b3, b4 )\
    FOR_5_FOR_3( macro, a1, a2, a3, a4, a5, b1, b2, b3 )\
    FOR_5_FOR_1( macro, a1, a2, a3, a4, a5, b4 )
#define FOR_5_FOR_5( macro, a1, a2, a3, a4, a5, b1, b2, b3, b4, b5 )\
    FOR_5_FOR_4( macro, a1, a2, a3, a4, a5, b1, b2, b3, b4 )\
    FOR_5_FOR_1( macro, a1, a2, a3, a4, a5, b5 )
#define FOR_5_FOR_6( macro, a1, a2, a3, a4, a5, b1, b2, b3, b4, b5, b6 )\
    FOR_5_FOR_5( macro, a1, a2, a3, a4, a5, b1, b2, b3, b4, b5 )\
    FOR_5_FOR_1( macro, a1, a2, a3, a4, a5, b6 )
#define FOR_5_FOR_7( macro, a1, a2, a3, a4, a5, b1, b2, b3, b4, b5, b6, b7 )\
    FOR_5_FOR_6( macro, a1, a2, a3, a4, a5, b1, b2, b3, b4, b5, b6 )\
    FOR_5_FOR_1( macro, a1, a2, a3, a4, a5, b7 )
#define FOR_5_FOR_8( macro, a1, a2, a3, a4, a5, b1, b2, b3, b4, b5, b6, b7, b8 )\
    FOR_5_FOR_7( macro, a1, a2, a3, a4, a5, b1, b2, b3, b4, b5, b6, b7 )\
    FOR_5_FOR_1( macro, a1, a2, a3, a4, a5, b8 )
#define FOR_5_FOR_9( macro, a1, a2, a3, a4, a5, b1, b2, b3, b4, b5, b6, b7, b8, b9 )\
    FOR_5_FOR_8( macro, a1, a2, a3, a4, a5, b1, b2, b3, b4, b5, b6, b7, b8 )\
    FOR_5_FOR_1( macro, a1, a2, a3, a4, a5, b9 )
#define FOR_5_FOR_10( macro, a1, a2, a3, a4, a5, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10 )\
    FOR_5_FOR_9( macro, a1, a2, a3, a4, a5, b1, b2, b3, b4, b5, b6, b7, b8, b9 )\
    FOR_5_FOR_1( macro, a1, a2, a3, a4, a5, b10 )

#define FOR_6_FOR_1( macro, a1, a2, a3, a4, a5, a6, b1 )\
    FOR_5_FOR_1( macro, a1, a2, a3, a4, a5, b1 )\
    FOR_1_FOR_1( macro, a6, b1 )
#define FOR_6_FOR_2( macro, a1, a2, a3, a4, a5, a6, b1, b2 )\
    FOR_6_FOR_1( macro, a1, a2, a3, a4, a5, a6, b1 )\
    FOR_6_FOR_1( macro, a1, a2, a3, a4, a5, a6, b2 )
#define FOR_6_FOR_3( macro, a1, a2, a3, a4, a5, a6, b1, b2, b3 )\
    FOR_6_FOR_2( macro, a1, a2, a3, a4, a5, a6, b1, b2 )\
    FOR_6_FOR_1( macro, a1, a2, a3, a4, a5, a6, b3 )
#define FOR_6_FOR_4( macro, a1, a2, a3, a4, a5, a6, b1, b2, b3, b4 )\
    FOR_6_FOR_3( macro, a1, a2, a3, a4, a5, a6, b1, b2, b3 )\
    FOR_6_FOR_1( macro, a1, a2, a3, a4, a5, a6, b4 )
#define FOR_6_FOR_5( macro, a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5 )\
    FOR_6_FOR_4( macro, a1, a2, a3, a4, a5, a6, b1, b2, b3, b4 )\
    FOR_6_FOR_1( macro, a1, a2, a3, a4, a5, a6, b5 )
#define FOR_6_FOR_6( macro, a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, b6 )\
    FOR_6_FOR_5( macro, a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5 )\
    FOR_6_FOR_1( macro, a1, a2, a3, a4, a5, a6, b6 )
#define FOR_6_FOR_7( macro, a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, b6, b7 )\
    FOR_6_FOR_6( macro, a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, b6 )\
    FOR_6_FOR_1( macro, a1, a2, a3, a4, a5, a6, b7 )
#define FOR_6_FOR_8( macro, a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, b6, b7, b8 )\
    FOR_6_FOR_7( macro, a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, b6, b7 )\
    FOR_6_FOR_1( macro, a1, a2, a3, a4, a5, a6, b8 )
#define FOR_6_FOR_9( macro, a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, b6, b7, b8, b9 )\
    FOR_6_FOR_8( macro, a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, b6, b7, b8 )\
    FOR_6_FOR_1( macro, a1, a2, a3, a4, a5, a6, b9 )
#define FOR_6_FOR_10( macro, a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10 )\
    FOR_6_FOR_9( macro, a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, b6, b7, b8, b9 )\
    FOR_6_FOR_1( macro, a1, a2, a3, a4, a5, a6, b10 )

#define FOR_7_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, b1 )\
    FOR_6_FOR_1( macro, a1, a2, a3, a4, a5, a6, b1 )\
    FOR_1_FOR_1( macro, a7, b1 )
#define FOR_7_FOR_2( macro, a1, a2, a3, a4, a5, a6, a7, b1, b2 )\
    FOR_7_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, b1 )\
    FOR_7_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, b2 )
#define FOR_7_FOR_3( macro, a1, a2, a3, a4, a5, a6, a7, b1, b2, b3 )\
    FOR_7_FOR_2( macro, a1, a2, a3, a4, a5, a6, a7, b1, b2 )\
    FOR_7_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, b3 )
#define FOR_7_FOR_4( macro, a1, a2, a3, a4, a5, a6, a7, b1, b2, b3, b4 )\
    FOR_7_FOR_3( macro, a1, a2, a3, a4, a5, a6, a7, b1, b2, b3 )\
    FOR_7_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, b4 )
#define FOR_7_FOR_5( macro, a1, a2, a3, a4, a5, a6, a7, b1, b2, b3, b4, b5 )\
    FOR_7_FOR_4( macro, a1, a2, a3, a4, a5, a6, a7, b1, b2, b3, b4 )\
    FOR_7_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, b5 )
#define FOR_7_FOR_6( macro, a1, a2, a3, a4, a5, a6, a7, b1, b2, b3, b4, b5, b6 )\
    FOR_7_FOR_5( macro, a1, a2, a3, a4, a5, a6, a7, b1, b2, b3, b4, b5 )\
    FOR_7_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, b6 )
#define FOR_7_FOR_7( macro, a1, a2, a3, a4, a5, a6, a7, b1, b2, b3, b4, b5, b6, b7 )\
    FOR_7_FOR_6( macro, a1, a2, a3, a4, a5, a6, a7, b1, b2, b3, b4, b5, b6 )\
    FOR_7_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, b7 )
#define FOR_7_FOR_8( macro, a1, a2, a3, a4, a5, a6, a7, b1, b2, b3, b4, b5, b6, b7, b8 )\
    FOR_7_FOR_7( macro, a1, a2, a3, a4, a5, a6, a7, b1, b2, b3, b4, b5, b6, b7 )\
    FOR_7_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, b8 )
#define FOR_7_FOR_9( macro, a1, a2, a3, a4, a5, a6, a7, b1, b2, b3, b4, b5, b6, b7, b8, b9 )\
    FOR_7_FOR_8( macro, a1, a2, a3, a4, a5, a6, a7, b1, b2, b3, b4, b5, b6, b7, b8 )\
    FOR_7_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, b9 )
#define FOR_7_FOR_10( macro, a1, a2, a3, a4, a5, a6, a7, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10 )\
    FOR_7_FOR_9( macro, a1, a2, a3, a4, a5, a6, a7, b1, b2, b3, b4, b5, b6, b7, b8, b9 )\
    FOR_7_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, b10 )

#define FOR_8_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, b1 )\
    FOR_7_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, b1 )\
    FOR_1_FOR_1( macro, a8, b1 )
#define FOR_8_FOR_2( macro, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2 )\
    FOR_8_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, b1 )\
    FOR_8_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, b2 )
#define FOR_8_FOR_3( macro, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, b3 )\
    FOR_8_FOR_2( macro, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2 )\
    FOR_8_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, b3 )
#define FOR_8_FOR_4( macro, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, b3, b4 )\
    FOR_8_FOR_3( macro, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, b3 )\
    FOR_8_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, b4 )
#define FOR_8_FOR_5( macro, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, b3, b4, b5 )\
    FOR_8_FOR_4( macro, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, b3, b4 )\
    FOR_8_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, b5 )
#define FOR_8_FOR_6( macro, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, b3, b4, b5, b6 )\
    FOR_8_FOR_5( macro, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, b3, b4, b5 )\
    FOR_8_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, b6 )
#define FOR_8_FOR_7( macro, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, b3, b4, b5, b6, b7 )\
    FOR_8_FOR_6( macro, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, b3, b4, b5, b6 )\
    FOR_8_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, b7 )
#define FOR_8_FOR_8( macro, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, b3, b4, b5, b6, b7, b8 )\
    FOR_8_FOR_7( macro, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, b3, b4, b5, b6, b7 )\
    FOR_8_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, b8 )
#define FOR_8_FOR_9( macro, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, b3, b4, b5, b6, b7, b8, b9 )\
    FOR_8_FOR_8( macro, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, b3, b4, b5, b6, b7, b8 )\
    FOR_8_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, b9 )
#define FOR_8_FOR_10( macro, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10 )\
    FOR_8_FOR_9( macro, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, b3, b4, b5, b6, b7, b8, b9 )\
    FOR_8_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, b10 )

#define FOR_9_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, b1 )\
    FOR_8_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, b1 )\
    FOR_1_FOR_1( macro, a9, b1 )
#define FOR_9_FOR_2( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, b1, b2 )\
    FOR_9_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, b1 )\
    FOR_9_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, b2 )
#define FOR_9_FOR_3( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, b1, b2, b3 )\
    FOR_9_FOR_2( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, b1, b2 )\
    FOR_9_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, b3 )
#define FOR_9_FOR_4( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, b1, b2, b3, b4 )\
    FOR_9_FOR_3( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, b1, b2, b3 )\
    FOR_9_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, b4 )
#define FOR_9_FOR_5( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, b1, b2, b3, b4, b5 )\
    FOR_9_FOR_4( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, b1, b2, b3, b4 )\
    FOR_9_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, b5 )
#define FOR_9_FOR_6( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, b1, b2, b3, b4, b5, b6 )\
    FOR_9_FOR_5( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, b1, b2, b3, b4, b5 )\
    FOR_9_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, b6 )
#define FOR_9_FOR_7( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, b1, b2, b3, b4, b5, b6, b7 )\
    FOR_9_FOR_6( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, b1, b2, b3, b4, b5, b6 )\
    FOR_9_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, b7 )
#define FOR_9_FOR_8( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, b1, b2, b3, b4, b5, b6, b7, b8 )\
    FOR_9_FOR_7( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, b1, b2, b3, b4, b5, b6, b7 )\
    FOR_9_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, b8 )
#define FOR_9_FOR_9( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, b1, b2, b3, b4, b5, b6, b7, b8, b9 )\
    FOR_9_FOR_8( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, b1, b2, b3, b4, b5, b6, b7, b8 )\
    FOR_9_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, b9 )
#define FOR_9_FOR_10( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10 )\
    FOR_9_FOR_9( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, b1, b2, b3, b4, b5, b6, b7, b8, b9 )\
    FOR_9_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, b10 )

#define FOR_10_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, b1 )\
    FOR_9_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, b1 )\
    FOR_1_FOR_1( macro, a10, b1 )
#define FOR_10_FOR_2( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, b1, b2 )\
    FOR_10_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, b1 )\
    FOR_10_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, b2 )
#define FOR_10_FOR_3( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, b1, b2, b3 )\
    FOR_10_FOR_2( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, b1, b2 )\
    FOR_10_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, b3 )
#define FOR_10_FOR_4( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, b1, b2, b3, b4 )\
    FOR_10_FOR_3( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, b1, b2, b3 )\
    FOR_10_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, b4 )
#define FOR_10_FOR_5( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, b1, b2, b3, b4, b5 )\
    FOR_10_FOR_4( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, b1, b2, b3, b4 )\
    FOR_10_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, b5 )
#define FOR_10_FOR_6( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, b1, b2, b3, b4, b5, b6 )\
    FOR_10_FOR_5( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, b1, b2, b3, b4, b5 )\
    FOR_10_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, b6 )
#define FOR_10_FOR_7( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, b1, b2, b3, b4, b5, b6, b7 )\
    FOR_10_FOR_6( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, b1, b2, b3, b4, b5, b6 )\
    FOR_10_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, b7 )
#define FOR_10_FOR_8( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, b1, b2, b3, b4, b5, b6, b7, b8 )\
    FOR_10_FOR_7( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, b1, b2, b3, b4, b5, b6, b7 )\
    FOR_10_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, b8 )
#define FOR_10_FOR_9( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, b1, b2, b3, b4, b5, b6, b7, b8, b9 )\
    FOR_10_FOR_8( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, b1, b2, b3, b4, b5, b6, b7, b8 )\
    FOR_10_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, b9 )
#define FOR_10_FOR_10( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10 )\
    FOR_10_FOR_9( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, b1, b2, b3, b4, b5, b6, b7, b8, b9 )\
    FOR_10_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, b10 )

#define FOR_11_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, b1 )\
    FOR_10_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, b1 )\
    FOR_1_FOR_1( macro, a11, b1 )
#define FOR_11_FOR_2( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, b1, b2 )\
    FOR_11_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, b1 )\
    FOR_11_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, b2 )
#define FOR_11_FOR_3( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, b1, b2, b3 )\
    FOR_11_FOR_2( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, b1, b2 )\
    FOR_11_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, b3 )
#define FOR_11_FOR_4( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, b1, b2, b3, b4 )\
    FOR_11_FOR_3( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, b1, b2, b3 )\
    FOR_11_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, b4 )
#define FOR_11_FOR_5( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, b1, b2, b3, b4, b5 )\
    FOR_11_FOR_4( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, b1, b2, b3, b4 )\
    FOR_11_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, b5 )
#define FOR_11_FOR_6( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, b1, b2, b3, b4, b5, b6 )\
    FOR_11_FOR_5( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, b1, b2, b3, b4, b5 )\
    FOR_11_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, b6 )
#define FOR_11_FOR_7( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, b1, b2, b3, b4, b5, b6, b7 )\
    FOR_11_FOR_6( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, b1, b2, b3, b4, b5, b6 )\
    FOR_11_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, b7 )
#define FOR_11_FOR_8( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, b1, b2, b3, b4, b5, b6, b7, b8 )\
    FOR_11_FOR_7( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, b1, b2, b3, b4, b5, b6, b7 )\
    FOR_11_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, b8 )
#define FOR_11_FOR_9( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, b1, b2, b3, b4, b5, b6, b7, b8, b9 )\
    FOR_11_FOR_8( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, b1, b2, b3, b4, b5, b6, b7, b8 )\
    FOR_11_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, b9 )
#define FOR_11_FOR_10( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10 )\
    FOR_11_FOR_9( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, b1, b2, b3, b4, b5, b6, b7, b8, b9 )\
    FOR_11_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, b10 )

#define FOR_12_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, b1 )\
    FOR_11_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, b1 )\
    FOR_1_FOR_1( macro, a12, b1 )
#define FOR_12_FOR_2( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, b1, b2 )\
    FOR_12_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, b1 )\
    FOR_12_FOR_1( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, b2 )
#define FOR_12_FOR_3( macro, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, b1, b2, b3 )\

#define FOR_1_FOR_1_FOR_1( macro, a1, b1, c1 )\
    macro( a1, b1, c1 )
#define FOR_1_FOR_1_FOR_2( macro, a1, b1, c1, c2 )\
    FOR_1_FOR_1_FOR_1( macro, a1, b1, c1 )\
    FOR_1_FOR_1_FOR_1( macro, a1, b1, c2 )
#define FOR_1_FOR_1_FOR_3( macro, a1, b1, c1, c2, c3 )\
    FOR_1_FOR_1_FOR_2( macro, a1, b1, c1, c2 )\
    FOR_1_FOR_1_FOR_1( macro, a1, b1, c3 )
#define FOR_1_FOR_1_FOR_4( macro, a1, b1, c1, c2, c3, c4 )\
    FOR_1_FOR_1_FOR_3( macro, a1, b1, c1, c2, c3 )\
    FOR_1_FOR_1_FOR_1( macro, a1, b1, c4 )
#define FOR_1_FOR_1_FOR_5( macro, a1, b1, c1, c2, c3, c4, c5 )\
    FOR_1_FOR_1_FOR_4( macro, a1, b1, c1, c2, c3, c4 )\
    FOR_1_FOR_1_FOR_1( macro, a1, b1, c5 )
#define FOR_1_FOR_1_FOR_6( macro, a1, b1, c1, c2, c3, c4, c5, c6 )\
    FOR_1_FOR_1_FOR_5( macro, a1, b1, c1, c2, c3, c4, c5 )\
    FOR_1_FOR_1_FOR_1( macro, a1, b1, c6 )
#define FOR_1_FOR_1_FOR_7( macro, a1, b1, c1, c2, c3, c4, c5, c6, c7 )\
    FOR_1_FOR_1_FOR_6( macro, a1, b1, c1, c2, c3, c4, c5, c6 )\
    FOR_1_FOR_1_FOR_1( macro, a1, b1, c7 )
#define FOR_1_FOR_1_FOR_8( macro, a1, b1, c1, c2, c3, c4, c5, c6, c7, c8 )\
    FOR_1_FOR_1_FOR_7( macro, a1, b1, c1, c2, c3, c4, c5, c6, c7 )\
    FOR_1_FOR_1_FOR_1( macro, a1, b1, c8 )
#define FOR_1_FOR_1_FOR_9( macro, a1, b1, c1, c2, c3, c4, c5, c6, c7, c8, c9 )\
    FOR_1_FOR_1_FOR_8( macro, a1, b1, c1, c2, c3, c4, c5, c6, c7, c8 )\
    FOR_1_FOR_1_FOR_1( macro, a1, b1, c9 )
#define FOR_1_FOR_1_FOR_10( macro, a1, b1, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10 )\
    FOR_1_FOR_1_FOR_9( macro, a1, b1, c1, c2, c3, c4, c5, c6, c7, c8, c9 )\
    FOR_1_FOR_1_FOR_1( macro, a1, b1, c10 )

#define FOR_1_FOR_2_FOR_1( macro, a1, b1, b2, c1 )\
    FOR_1_FOR_1_FOR_1( macro, a1, b1, c1 )\
    FOR_1_FOR_1_FOR_1( macro, a1, b2, c1 )
#define FOR_1_FOR_2_FOR_2( macro, a1, b1, b2, c1, c2 )\
    FOR_1_FOR_2_FOR_1( macro, a1, b1, b2, c1 )\
    FOR_1_FOR_2_FOR_1( macro, a1, b1, b2, c2 )
#define FOR_1_FOR_2_FOR_3( macro, a1, b1, b2, c1, c2, c3 )\
    FOR_1_FOR_2_FOR_2( macro, a1, b1, b2, c1, c2 )\
    FOR_1_FOR_2_FOR_1( macro, a1, b1, b2, c3 )
#define FOR_1_FOR_2_FOR_4( macro, a1, b1, b2, c1, c2, c3, c4 )\
    FOR_1_FOR_2_FOR_3( macro, a1, b1, b2, c1, c2, c3 )\
    FOR_1_FOR_2_FOR_1( macro, a1, b1, b2, c4 )
#define FOR_1_FOR_2_FOR_5( macro, a1, b1, b2, c1, c2, c3, c4, c5 )\
    FOR_1_FOR_2_FOR_4( macro, a1, b1, b2, c1, c2, c3, c4 )\
    FOR_1_FOR_2_FOR_1( macro, a1, b1, b2, c5 )
#define FOR_1_FOR_2_FOR_6( macro, a1, b1, b2, c1, c2, c3, c4, c5, c6 )\
    FOR_1_FOR_2_FOR_5( macro, a1, b1, b2, c1, c2, c3, c4, c5 )\
    FOR_1_FOR_2_FOR_1( macro, a1, b1, b2, c6 )
#define FOR_1_FOR_2_FOR_7( macro, a1, b1, b2, c1, c2, c3, c4, c5, c6, c7 )\
    FOR_1_FOR_2_FOR_6( macro, a1, b1, b2, c1, c2, c3, c4, c5, c6 )\
    FOR_1_FOR_2_FOR_1( macro, a1, b1, b2, c7 )
#define FOR_1_FOR_2_FOR_8( macro, a1, b1, b2, c1, c2, c3, c4, c5, c6, c7, c8 )\
    FOR_1_FOR_2_FOR_7( macro, a1, b1, b2, c1, c2, c3, c4, c5, c6, c7 )\
    FOR_1_FOR_2_FOR_1( macro, a1, b1, b2, c8 )
#define FOR_1_FOR_2_FOR_9( macro, a1, b1, b2, c1, c2, c3, c4, c5, c6, c7, c8, c9 )\
    FOR_1_FOR_2_FOR_8( macro, a1, b1, b2, c1, c2, c3, c4, c5, c6, c7, c8 )\
    FOR_1_FOR_2_FOR_1( macro, a1, b1, b2, c9 )
#define FOR_1_FOR_2_FOR_10( macro, a1, b1, b2, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10 )\
    FOR_1_FOR_2_FOR_9( macro, a1, b1, b2, c1, c2, c3, c4, c5, c6, c7, c8, c9 )\
    FOR_1_FOR_2_FOR_1( macro, a1, b1, b2, c10 )

#define FOR_1_FOR_3_FOR_1( macro, a1, b1, b2, b3, c1 )\
    FOR_1_FOR_2_FOR_1( macro, a1, b1, b2, c1 )\
    FOR_1_FOR_2_FOR_1( macro, a1, b3, b2, c1 )
#define FOR_1_FOR_3_FOR_2( macro, a1, b1, b2, b3, c1, c2 )\
    FOR_1_FOR_3_FOR_1( macro, a1, b1, b2, b3, c1 )\
    FOR_1_FOR_3_FOR_1( macro, a1, b1, b2, b3, c2 )
#define FOR_1_FOR_3_FOR_3( macro, a1, b1, b2, b3, c1, c2, c3 )\
    FOR_1_FOR_3_FOR_2( macro, a1, b1, b2, b3, c1, c2 )\
    FOR_1_FOR_3_FOR_1( macro, a1, b1, b2, b3, c3 )
#define FOR_1_FOR_3_FOR_4( macro, a1, b1, b2, b3, c1, c2, c3, c4 )\
    FOR_1_FOR_3_FOR_3( macro, a1, b1, b2, b3, c1, c2, c3 )\
    FOR_1_FOR_3_FOR_1( macro, a1, b1, b2, b3, c4 )
#define FOR_1_FOR_3_FOR_5( macro, a1, b1, b2, b3, c1, c2, c3, c4, c5 )\
    FOR_1_FOR_3_FOR_4( macro, a1, b1, b2, b3, c1, c2, c3, c4 )\
    FOR_1_FOR_3_FOR_1( macro, a1, b1, b2, b3, c5 )
#define FOR_1_FOR_3_FOR_6( macro, a1, b1, b2, b3, c1, c2, c3, c4, c5, c6 )\
    FOR_1_FOR_3_FOR_5( macro, a1, b1, b2, b3, c1, c2, c3, c4, c5 )\
    FOR_1_FOR_3_FOR_1( macro, a1, b1, b2, b3, c6 )
#define FOR_1_FOR_3_FOR_7( macro, a1, b1, b2, b3, c1, c2, c3, c4, c5, c6, c7 )\
    FOR_1_FOR_3_FOR_6( macro, a1, b1, b2, b3, c1, c2, c3, c4, c5, c6 )\
    FOR_1_FOR_3_FOR_1( macro, a1, b1, b2, b3, c7 )
#define FOR_1_FOR_3_FOR_8( macro, a1, b1, b2, b3, c1, c2, c3, c4, c5, c6, c7, c8 )\
    FOR_1_FOR_3_FOR_7( macro, a1, b1, b2, b3, c1, c2, c3, c4, c5, c6, c7 )\
    FOR_1_FOR_3_FOR_1( macro, a1, b1, b2, b3, c8 )
#define FOR_1_FOR_3_FOR_9( macro, a1, b1, b2, b3, c1, c2, c3, c4, c5, c6, c7, c8, c9 )\
    FOR_1_FOR_3_FOR_8( macro, a1, b1, b2, b3, c1, c2, c3, c4, c5, c6, c7, c8 )\
    FOR_1_FOR_3_FOR_1( macro, a1, b1, b2, b3, c9 )
#define FOR_1_FOR_3_FOR_10( macro, a1, b1, b2, b3, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10 )\
    FOR_1_FOR_3_FOR_9( macro, a1, b1, b2, b3, c1, c2, c3, c4, c5, c6, c7, c8, c9 )\
    FOR_1_FOR_3_FOR_1( macro, a1, b1, b2, b3, c10 )

#define FOR_1_FOR_4_FOR_1( macro, a1, b1, b2, b3, b4, c1 )\
    FOR_1_FOR_3_FOR_1( macro, a1, b1, b2, b3, c1 )\
    FOR_1_FOR_3_FOR_1( macro, a1, b4, b2, b3, c1 )
#define FOR_1_FOR_4_FOR_2( macro, a1, b1, b2, b3, b4, c1, c2 )\
    FOR_1_FOR_4_FOR_1( macro, a1, b1, b2, b3, b4, c1 )\
    FOR_1_FOR_4_FOR_1( macro, a1, b1, b2, b3, b4, c2 )
#define FOR_1_FOR_4_FOR_3( macro, a1, b1, b2, b3, b4, c1, c2, c3 )\
    FOR_1_FOR_4_FOR_2( macro, a1, b1, b2, b3, b4, c1, c2 )\
    FOR_1_FOR_4_FOR_1( macro, a1, b1, b2, b3, b4, c3 )
#define FOR_1_FOR_4_FOR_4( macro, a1, b1, b2, b3, b4, c1, c2, c3, c4 )\
    FOR_1_FOR_4_FOR_3( macro, a1, b1, b2, b3, b4, c1, c2, c3 )\
    FOR_1_FOR_4_FOR_1( macro, a1, b1, b2, b3, b4, c4 )
#define FOR_1_FOR_4_FOR_5( macro, a1, b1, b2, b3, b4, c1, c2, c3, c4, c5 )\
    FOR_1_FOR_4_FOR_4( macro, a1, b1, b2, b3, b4, c1, c2, c3, c4 )\
    FOR_1_FOR_4_FOR_1( macro, a1, b1, b2, b3, b4, c5 )
#define FOR_1_FOR_4_FOR_6( macro, a1, b1, b2, b3, b4, c1, c2, c3, c4, c5, c6 )\
    FOR_1_FOR_4_FOR_5( macro, a1, b1, b2, b3, b4, c1, c2, c3, c4, c5 )\
    FOR_1_FOR_4_FOR_1( macro, a1, b1, b2, b3, b4, c6 )
#define FOR_1_FOR_4_FOR_7( macro, a1, b1, b2, b3, b4, c1, c2, c3, c4, c5, c6, c7 )\
    FOR_1_FOR_4_FOR_6( macro, a1, b1, b2, b3, b4, c1, c2, c3, c4, c5, c6 )\
    FOR_1_FOR_4_FOR_1( macro, a1, b1, b2, b3, b4, c7 )
#define FOR_1_FOR_4_FOR_8( macro, a1, b1, b2, b3, b4, c1, c2, c3, c4, c5, c6, c7, c8 )\
    FOR_1_FOR_4_FOR_7( macro, a1, b1, b2, b3, b4, c1, c2, c3, c4, c5, c6, c7 )\
    FOR_1_FOR_4_FOR_1( macro, a1, b1, b2, b3, b4, c8 )
#define FOR_1_FOR_4_FOR_9( macro, a1, b1, b2, b3, b4, c1, c2, c3, c4, c5, c6, c7, c8, c9 )\
    FOR_1_FOR_4_FOR_8( macro, a1, b1, b2, b3, b4, c1, c2, c3, c4, c5, c6, c7, c8 )\
    FOR_1_FOR_4_FOR_1( macro, a1, b1, b2, b3, b4, c9 )
#define FOR_1_FOR_4_FOR_10( macro, a1, b1, b2, b3, b4, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10 )\
    FOR_1_FOR_4_FOR_9( macro, a1, b1, b2, b3, b4, c1, c2, c3, c4, c5, c6, c7, c8, c9 )\
    FOR_1_FOR_4_FOR_1( macro, a1, b1, b2, b3, b4, c10 )

#define FOR_1_FOR_5_FOR_1( macro, a1, b1, b2, b3, b4, b5, c1 )\
    FOR_1_FOR_4_FOR_1( macro, a1, b1, b2, b3, b4, c1 )\
    FOR_1_FOR_4_FOR_1( macro, a1, b5, b2, b3, b4, c1 )
#define FOR_1_FOR_5_FOR_2( macro, a1, b1, b2, b3, b4, b5, c1, c2 )\
    FOR_1_FOR_5_FOR_1( macro, a1, b1, b2, b3, b4, b5, c1 )\
    FOR_1_FOR_5_FOR_1( macro, a1, b1, b2, b3, b4, b5, c2 )
#define FOR_1_FOR_5_FOR_3( macro, a1, b1, b2, b3, b4, b5, c1, c2, c3 )\
    FOR_1_FOR_5_FOR_2( macro, a1, b1, b2, b3, b4, b5, c1, c2 )\
    FOR_1_FOR_5_FOR_1( macro, a1, b1, b2, b3, b4, b5, c3 )
#define FOR_1_FOR_5_FOR_4( macro, a1, b1, b2, b3, b4, b5, c1, c2, c3, c4 )\
    FOR_1_FOR_5_FOR_3( macro, a1, b1, b2, b3, b4, b5, c1, c2, c3 )\
    FOR_1_FOR_5_FOR_1( macro, a1, b1, b2, b3, b4, b5, c4 )
#define FOR_1_FOR_5_FOR_5( macro, a1, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5 )\
    FOR_1_FOR_5_FOR_4( macro, a1, b1, b2, b3, b4, b5, c1, c2, c3, c4 )\
    FOR_1_FOR_5_FOR_1( macro, a1, b1, b2, b3, b4, b5, c5 )
#define FOR_1_FOR_5_FOR_6( macro, a1, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, c6 )\
    FOR_1_FOR_5_FOR_5( macro, a1, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5 )\
    FOR_1_FOR_5_FOR_1( macro, a1, b1, b2, b3, b4, b5, c6 )
#define FOR_1_FOR_5_FOR_7( macro, a1, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, c6, c7 )\
    FOR_1_FOR_5_FOR_6( macro, a1, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, c6 )\
    FOR_1_FOR_5_FOR_1( macro, a1, b1, b2, b3, b4, b5, c7 )
#define FOR_1_FOR_5_FOR_8( macro, a1, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, c6, c7, c8 )\
    FOR_1_FOR_5_FOR_7( macro, a1, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, c6, c7 )\
    FOR_1_FOR_5_FOR_1( macro, a1, b1, b2, b3, b4, b5, c8 )
#define FOR_1_FOR_5_FOR_9( macro, a1, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, c6, c7, c8, c9 )\
    FOR_1_FOR_5_FOR_8( macro, a1, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, c6, c7, c8 )\
    FOR_1_FOR_5_FOR_1( macro, a1, b1, b2, b3, b4, b5, c9 )
#define FOR_1_FOR_5_FOR_10( macro, a1, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10 )\
    FOR_1_FOR_5_FOR_9( macro, a1, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, c6, c7, c8, c9 )\
    FOR_1_FOR_5_FOR_1( macro, a1, b1, b2, b3, b4, b5, c10 )

#endif // BEWUSSTSEIN_NEURAL_NETWORKS_CPP_MACRO_FOR_HPP_
