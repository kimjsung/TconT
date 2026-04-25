#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <regex>

template <typename T>
void pre_Initializing_Input_Tensors(T* A, int size_A, T* B, int size_B)
{
    srand(time(NULL));

	int i, j;
	for (i = 0; i < size_A; i++)
	{
		A[i] = static_cast<T>(static_cast<double>(rand()) / RAND_MAX);
	}

	for (j = 0; j < size_B; j++)
	{
		B[j] = static_cast<T>(static_cast<double>(rand()) / RAND_MAX);
	}
}

void create_vector(int equation_num, std::vector<int>& modeC, std::vector<int>& modeA, std::vector<int>& modeB, std::string& op) {
    switch(equation_num) {
        case 1 :
            modeC = {'a','b','c'};
            modeA = {'b','d','a'};
            modeB = {'d','c'};
            op = "+=";
            break;
        case 2 :
            modeC = {'a','b','c'};
            modeA = {'d','c','a'};
            modeB = {'b','d'};
            op = "+=";
            break;
        case 3 :
            modeC = {'a','b','c','d'};
            modeA = {'d','b','e','a'};
            modeB = {'e','c'};
            op = "+=";
            break;
        case 4 :
            modeC = {'a','b','c','d'};
            modeA = {'d','e','c','a'};
            modeB = {'b','e'};
            op = "+=";
            break;
        case 5 :
            modeC = {'a','b','c','d'};
            modeA = {'e','b','a','d'};
            modeB = {'c','e'};
            op = "+=";
            break;
        case 6 :
            modeC = {'a','b','c','d','e'};
            modeA = {'e','f','b','a','d'};
            modeB = {'c','f'};
            op = "+=";
            break;
        case 7 :
            modeC = {'a','b','c','d','e'};
            modeA = {'e','c','b','f','a'};
            modeB = {'f','d'};
            op = "+=";
            break;
        case 8 :
            modeC = {'a','b','c','d','e'};
            modeA = {'e','f','c','a','d'};
            modeB = {'b','f'};
            op = "+=";
            break;
        case 9 :
            modeC = {'a','b','c','d'};
            modeA = {'e','a'};
            modeB = {'e','b','c','d'};
            op = "+=";
            break;
        case 10 :
            modeC = {'a','b','c','d'};
            modeA = {'e','b'};
            modeB = {'a','e','c','d'};
            op = "+=";
            break;
        case 11 :
            modeC = {'a','b','c','d'};
            modeA = {'e','c'};
            modeB = {'a','b','e','d'};
            op = "+=";
            break;
        case 12 :
            modeC = {'a','b'};
            modeA = {'a','c'};
            modeB = {'c','b'};
            op = "+=";
            break;
        case 13 :
            modeC = {'a','b'};
            modeA = {'a','c','d'};
            modeB = {'d','b','c'};
            op = "+=";
            break;
        case 14 :
            modeC = {'a','b'};
            modeA = {'c','a','d'};
            modeB = {'d','c','b'};
            op = "+=";
            break;
        case 15 :
            modeC = {'a','b','c'};
            modeA = {'a','c','d'};
            modeB = {'d','b'};
            op = "+=";
            break;
        case 16 :
            modeC = {'a','b','c'};
            modeA = {'a','d'};
            modeB = {'b','d','c'};
            op = "+=";
            break;
        case 17 :
            modeC = {'a','b','c'};
            modeA = {'a','d','c'};
            modeB = {'b','d'};
            op = "+=";
            break;
        case 18 :
            modeC = {'a','b','c'};
            modeA = {'a','d','c'};
            modeB = {'d','b'};
            op = "+=";
            break;
        case 19 :
            modeC = {'a','b','c'};
            modeA = {'a','d','e','c'};
            modeB = {'e','b','d'};
            op = "+=";
            break;
        case 20 :
            modeC = {'a','b','c','d'};
            modeA = {'a','e','b','f'};
            modeB = {'d','f','c','e'};
            op = "+=";
            break;
        case 21 :
            modeC = {'a','b','c','d'};
            modeA = {'a','e','b','f'};
            modeB = {'f','d','e','c'};
            op = "+=";
            break;
        case 22 :
            modeC = {'a','b','c','d'};
            modeA = {'a','e','c','f'};
            modeB = {'b','f','d','e'};
            op = "+=";
            break;
        case 23 :
            modeC = {'a','b','c','d'};
            modeA = {'a','e','c','f'};
            modeB = {'f','b','e','d'};
            op = "+=";
            break;
        case 24 :
            modeC = {'a','b','c','d'};
            modeA = {'a','e','d','f'};
            modeB = {'b','f','c','e'};
            op = "+=";
            break;
        case 25 :
            modeC = {'a','b','c','d'};
            modeA = {'a','e','d','f'};
            modeB = {'f','b','e','c'};
            op = "+=";
            break;
        case 26 :
            modeC = {'a','b','c','d'};
            modeA = {'a','e','f','b'};
            modeB = {'f','d','c','e'};
            op = "+=";
            break;
        case 27 :
            modeC = {'a','b','c','d'};
            modeA = {'a','e','f','c'};
            modeB = {'f','b','e','d'};
            op = "+=";
            break;
        case 28 :
            modeC = {'a','b','c','d'};
            modeA = {'e','a','f','b'};
            modeB = {'f','d','e','c'};
            op = "+=";
            break;
        case 29 :
            modeC = {'a','b','c','d'};
            modeA = {'e','a','f','c'};
            modeB = {'b','f','d','e'};
            op = "+=";
            break;
        case 30 :
            modeC = {'a','b','c','d'};
            modeA = {'e','a','f','d'};
            modeB = {'f','b','e','c'};
            op = "+=";
            break;
        case 31 :
            modeC = {'a','b','c','d','e','f'};
            modeA = {'d','e','g','a'};
            modeB = {'g','f','b','c'};
            op = "-=";
            break;
        case 32 :
            modeC = {'a','b','c','d','e','f'};     
            modeA = {'d','e','g','b'};
            modeB = {'g','f','a','c'};
            op = "+=";
            break;
        case 33 :
            modeC = {'a','b','c','d','e','f'};
            modeA = {'d','e','g','c'};
            modeB = {'g','f','a','b'};
            op = "-=";
            break;
        case 34 :
            modeC = {'a','b','c','d','e','f'};
            modeA = {'d','f','g','a'};
            modeB = {'g','e','b','c'};
            op = "-=";
            break;
        case 35 :
            modeC = {'a','b','c','d','e','f'};
            modeA = {'d','f','g','b'};
            modeB = {'g','e','a','c'};
            op = "+=";
            break;
        case 36 :
            modeC = {'a','b','c','d','e','f'};
            modeA = {'d','f','g','c'};
            modeB = {'g','e','a','b'};
            op = "+=";
            break;
        case 37 :
            modeC = {'a','b','c','d','e','f'};
            modeA = {'e','f','g','a'};
            modeB = {'g','d','b','c'};
            op = "+=";
            break;
        case 38 :
            modeC = {'a','b','c','d','e','f'};
            modeA = {'e','f','g','b'};
            modeB = {'g','d','a','c'};
            op = "+=";
            break;
        case 39 :
            modeC = {'a','b','c','d','e','f'};
            modeA = {'e','f','g','c'};
            modeB = {'g','d','a','b'};
            op = "+=";
            break;
        case 40 :
            modeC = {'a','b','c','d','e','f'};
            modeA = {'g','d','a','b'};
            modeB = {'e','f','g','c'};
            op = "+=";
            break;
        case 41 :
            modeC = {'a','b','c','d','e','f'};
            modeA = {'g','d','a','c'};
            modeB = {'e','f','g','b'};
            op = "+=";
            break;
        case 42 :
            modeC = {'a','b','c','d','e','f'};
            modeA = {'g','d','b','c'};
            modeB = {'e','f','g','a'};
            op = "+=";
            break;
        case 43 :
            modeC = {'a','b','c','d','e','f'};
            modeA = {'g','e','a','b'};
            modeB = {'d','f','g','c'};
            op = "+=";
            break;
        case 44 :
            modeC = {'a','b','c','d','e','f'};
            modeA = {'g','e','a','c'};
            modeB = {'d','f','g','b'};
            op = "+=";
            break;
        case 45 :
            modeC = {'a','b','c','d','e','f'};
            modeA = {'g','e','b','c'};
            modeB = {'d','f','g','a'};
            op = "+=";
            break;
        case 46 :
            modeC = {'a','b','c','d','e','f'};
            modeA = {'g','f','a','b'};
            modeB = {'d','e','g','c'};
            op = "+=";
            break;
        case 47 :
            modeC = {'a','b','c','d','e','f'};
            modeA = {'g','f','a','c'};
            modeB = {'d','e','g','b'};
            op = "+=";
            break;
        case 48 :
            modeC = {'a','b','c','d','e','f'};
            modeA = {'g','f','b','c'};
            modeB = {'d','e','g','a'};
            op = "+=";
            break;
        default :
            printf("Invalid Equation Number\n");
            break;
    }

    return;
}
void set_vector_size_with_args(int equation_num, std::vector<int64_t>& extentC, std::vector<int64_t>& extentA, std::vector<int64_t>& extentB, std::unordered_map<int, int64_t>& extent, int argc, std::vector<std::string> argv) {
    if(equation_num < 3) {
        if (argc < 4) {
            std::cerr << "Usage: <program> <a> <b> <c> <d>" << std::endl;
            exit(1);
        }
        extent['a'] = std::stoll(argv[0]);
        extent['b'] = std::stoll(argv[1]);
        extent['c'] = std::stoll(argv[2]);
        extent['d'] = std::stoll(argv[3]);
    }
    else if(equation_num >= 3 && equation_num <= 5) {
        if (argc < 5) {
            std::cerr << "Usage: <program> <a> <b> <c> <d> <e>" << std::endl;
            exit(1);
        }
        extent['a'] = std::stoll(argv[0]);
        extent['b'] = std::stoll(argv[1]);
        extent['c'] = std::stoll(argv[2]);
        extent['d'] = std::stoll(argv[3]);
        extent['e'] = std::stoll(argv[4]);
    }
    else if(equation_num >= 6 && equation_num <= 8) {
        if (argc < 6) {
            std::cerr << "Usage: <program> <a> <b> <c> <d> <e> <f>" << std::endl;
            exit(1);
        }
        extent['a'] = std::stoll(argv[0]);
        extent['b'] = std::stoll(argv[1]);
        extent['c'] = std::stoll(argv[2]);
        extent['d'] = std::stoll(argv[3]);
        extent['e'] = std::stoll(argv[4]);
        extent['f'] = std::stoll(argv[5]);
    }
    else if(equation_num >= 9 && equation_num <= 11) {
        if (argc < 5) {
            std::cerr << "Usage: <program> <a> <b> <c> <d> <e>" << std::endl;
            exit(1);
        }
        extent['a'] = std::stoll(argv[0]);
        extent['b'] = std::stoll(argv[1]);
        extent['c'] = std::stoll(argv[2]);
        extent['d'] = std::stoll(argv[3]);
        extent['e'] = std::stoll(argv[4]);
    }
    else if(equation_num == 12) {
        if (argc < 3) {
            std::cerr << "Usage: <program> <a> <b> <c>" << std::endl;
            exit(1);
        }
        extent['a'] = std::stoll(argv[0]);
        extent['b'] = std::stoll(argv[1]);
        extent['c'] = std::stoll(argv[2]);
    }
    else if(equation_num >= 13 && equation_num <= 18) {
        if (argc < 4) {
            std::cerr << "Usage: <program> <a> <b> <c> <d>" << std::endl;
            exit(1);
        }
        extent['a'] = std::stoll(argv[0]);
        extent['b'] = std::stoll(argv[1]);
        extent['c'] = std::stoll(argv[2]);
        extent['d'] = std::stoll(argv[3]);
    }
    else if(equation_num == 19) {
        if (argc < 5) {
            std::cerr << "Usage: <program> <a> <b> <c> <d> <e>" << std::endl;
            exit(1);
        }
        extent['a'] = std::stoll(argv[0]);
        extent['b'] = std::stoll(argv[1]);
        extent['c'] = std::stoll(argv[2]);
        extent['d'] = std::stoll(argv[3]);
        extent['e'] = std::stoll(argv[4]);
    }
    else if(equation_num >= 20 && equation_num <= 30) {
        if (argc < 6) {
            std::cerr << "Usage: <program> <a> <b> <c> <d> <e> <f>" << std::endl;
            exit(1);
        }
        extent['a'] = std::stoll(argv[0]);
        extent['b'] = std::stoll(argv[1]);
        extent['c'] = std::stoll(argv[2]);
        extent['d'] = std::stoll(argv[3]);
        extent['e'] = std::stoll(argv[4]);
        extent['f'] = std::stoll(argv[5]);
    }
    else if(equation_num >= 31 && equation_num <= 48) {
        if (argc < 7) {
            std::cerr << "Usage: <program> <a> <b> <c> <d> <e> <f> <g>" << std::endl;
            exit(1);
        }
        extent['a'] = std::stoll(argv[0]);
        extent['b'] = std::stoll(argv[1]);
        extent['c'] = std::stoll(argv[2]);
        extent['d'] = std::stoll(argv[3]);
        extent['e'] = std::stoll(argv[4]);
        extent['f'] = std::stoll(argv[5]);
        extent['g'] = std::stoll(argv[6]);
    }
    switch(equation_num) {
        case 1 :
            extentC.clear();
            for(auto mode : {'a','b','c'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'b','d','a'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'d','c'})
                extentB.push_back(extent[mode]);
            break;
        case 2 :
            extentC.clear();
            for(auto mode : {'a','b','c'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'d','c','a'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'b','d'})
                extentB.push_back(extent[mode]);
            break;
        case 3 :
            extentC.clear();
            for(auto mode : {'a','b','c','d'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'d','b','e','a'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'e','c'})
                extentB.push_back(extent[mode]);
            break;
        case 4 :
            extentC.clear();
            for(auto mode : {'a','b','c','d'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'d','e','c','a'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'b','e'})
                extentB.push_back(extent[mode]);
            break;
        case 5 :
            extentC.clear();
            for(auto mode : {'a','b','c','d'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'e','b','a','d'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'c','e'})
                extentB.push_back(extent[mode]);
            break;
        case 6 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'e','f','b','a','d'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'c','f'})
                extentB.push_back(extent[mode]);
            break;
        case 7 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'e','c','b','f','a'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'f','d'})
                extentB.push_back(extent[mode]);
            break;
        case 8 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'e','f','c','a','d'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'b','f'})
                extentB.push_back(extent[mode]);
            break;
        case 9 :
            extentC.clear();
            for(auto mode : {'a','b','c','d'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'e','a'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'e','b','c','d'})
                extentB.push_back(extent[mode]);
            break;
        case 10 :
            extentC.clear();
            for(auto mode : {'a','b','c','d'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'e','b'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'a','e','c','d'})
                extentB.push_back(extent[mode]);
            break;
        case 11 :
            extentC.clear();
            for(auto mode : {'a','b','c','d'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'e','c'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'a','b','e','d'})
                extentB.push_back(extent[mode]);
            break;
        case 12 :
            extentC.clear();
            for(auto mode : {'a','b'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'a','c'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'c','b'})
                extentB.push_back(extent[mode]);
            break;
        case 13 :
            extentC.clear();
            for(auto mode : {'a','b'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'a','c','d'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'d','b','c'})
                extentB.push_back(extent[mode]);
            break;
        case 14 :
            extentC.clear();
            for(auto mode : {'a','b'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'c','a','d'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'d','c','b'})
                extentB.push_back(extent[mode]);
            break;
        case 15 :
            extentC.clear();
            for(auto mode : {'a','b','c'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'a','c','d'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'d','b'})
                extentB.push_back(extent[mode]);
            break;
        case 16 :
            extentC.clear();
            for(auto mode : {'a','b','c'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'a','d'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'b','d','c'})
                extentB.push_back(extent[mode]);
            break;
        case 17 :
            extentC.clear();
            for(auto mode : {'a','b','c'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'a','d','c'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'b','d'})
                extentB.push_back(extent[mode]);
            break;
        case 18 :
            extentC.clear();
            for(auto mode : {'a','b','c'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'a','d','c'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'d','b'})
                extentB.push_back(extent[mode]);
            break;
        case 19 :
            extentC.clear();
            for(auto mode : {'a','b','c'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'a','d','e','c'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'e','b','d'})
                extentB.push_back(extent[mode]);
            break;
        case 20 :
            extentC.clear();
            for(auto mode : {'a','b','c','d'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'a','e','b','f'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'d','f','c','e'})
                extentB.push_back(extent[mode]);
            break;
        case 21 :
            extentC.clear();
            for(auto mode : {'a','b','c','d'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'a','e','b','f'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'f','d','e','c'})
                extentB.push_back(extent[mode]);
            break;
        case 22 :
            extentC.clear();
            for(auto mode : {'a','b','c','d'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'a','e','c','f'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'b','f','d','e'})
                extentB.push_back(extent[mode]);
            break;
        case 23 :
            extentC.clear();
            for(auto mode : {'a','b','c','d'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'a','e','c','f'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'f','b','e','d'})
                extentB.push_back(extent[mode]);
            break;
        case 24 :
            extentC.clear();
            for(auto mode : {'a','b','c','d'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'a','e','d','f'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'b','f','c','e'})
                extentB.push_back(extent[mode]);
            break;
        case 25 :
            extentC.clear();
            for(auto mode : {'a','b','c','d'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'a','e','d','f'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'f','b','e','c'})
                extentB.push_back(extent[mode]);
            break;
        case 26 :
            extentC.clear();
            for(auto mode : {'a','b','c','d'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'a','e','f','b'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'f','d','c','e'})
                extentB.push_back(extent[mode]);
            break;
        case 27 :
            extentC.clear();
            for(auto mode : {'a','b','c','d'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'a','e','f','c'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'f','b','e','d'})
                extentB.push_back(extent[mode]);
            break;
        case 28 :
            extentC.clear();
            for(auto mode : {'a','b','c','d'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'e','a','f','b'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'f','d','e','c'})
                extentB.push_back(extent[mode]);
            break;
        case 29 :
            extentC.clear();
            for(auto mode : {'a','b','c','d'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'e','a','f','c'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'b','f','d','e'})
                extentB.push_back(extent[mode]);
            break;
        case 30 :
            extentC.clear();
            for(auto mode : {'a','b','c','d'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'e','a','f','d'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'f','b','e','c'})
                extentB.push_back(extent[mode]);
            break;
        case 31 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e','f'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'d','e','g','a'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'g','f','b','c'})
                extentB.push_back(extent[mode]);
            break;
        case 32 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e','f'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'d','e','g','b'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'g','f','a','c'})
                extentB.push_back(extent[mode]);
            break;
        case 33 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e','f'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'d','e','g','c'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'g','f','a','b'})
                extentB.push_back(extent[mode]);
            break;
        case 34 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e','f'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'d','f','g','a'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'g','e','b','c'})
                extentB.push_back(extent[mode]);
            break;
        case 35 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e','f'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'d','f','g','b'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'g','e','a','c'})
                extentB.push_back(extent[mode]);
            break;
        case 36 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e','f'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'d','f','g','c'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'g','e','a','b'})
                extentB.push_back(extent[mode]);
            break;
        case 37 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e','f'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'e','f','g','a'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'g','d','b','c'})
                extentB.push_back(extent[mode]);
            break;
        case 38 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e','f'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'e','f','g','b'})
                extentA.push_back(extent[mode]);
                
            extentB.clear();
            for(auto mode : {'g','d','a','c'})
                extentB.push_back(extent[mode]);
            break;
        case 39 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e','f'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'e','f','g','c'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'g','d','a','b'})
                extentB.push_back(extent[mode]);
            break;
        case 40 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e','f'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'g','d','a','b'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'e','f','g','c'})
                extentB.push_back(extent[mode]);
            break;
        case 41 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e','f'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'g','d','a','c'})
                extentA.push_back(extent[mode]);   
            extentB.clear();
            for(auto mode : {'e','f','g','b'})
                extentB.push_back(extent[mode]);
            break;
        case 42 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e','f'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'g','d','b','c'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'e','f','g','a'})
                extentB.push_back(extent[mode]);
            break;
        case 43 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e','f'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'g','e','a','b'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'d','f','g','c'})
                extentB.push_back(extent[mode]);
            break;
        case 44 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e','f'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'g','e','a','c'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'d','f','g','b'})
                extentB.push_back(extent[mode]);
            break;
        case 45 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e','f'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'g','e','b','c'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'d','f','g','a'})
                extentB.push_back(extent[mode]);
            break;
        case 46 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e','f'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'g','f','a','b'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'d','e','g','c'})
                extentB.push_back(extent[mode]);
            break;
        case 47 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e','f'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'g','f','a','c'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'d','e','g','b'})
                extentB.push_back(extent[mode]);
            break;
        case 48 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e','f'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'g','f','b','c'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'d','e','g','a'})
                extentB.push_back(extent[mode]);
            break;
        default :
            printf("Invalid Equation Number\n");
            break;
    }
    
    return;
}
void set_vector_size(int equation_num, std::vector<int64_t>& extentC, std::vector<int64_t>& extentA, std::vector<int64_t>& extentB, std::unordered_map<int, int64_t>& extent) {
    if(equation_num == 1) {
        extent['a'] = 312;
        extent['b'] = 312;
        extent['c'] = 24;
        extent['d'] = 312;
    }
    else if(equation_num == 2) {
        extent['a'] = 312;
        extent['b'] = 24;
        extent['c'] = 296;
        extent['d'] = 312;
    }
    else if(equation_num == 3) {
        extent['a'] = 72;
        extent['b'] = 72;
        extent['c'] = 24;
        extent['d'] = 72;
        extent['e'] = 72;
    }
    else if(equation_num == 4) {
        extent['a'] = 72;
        extent['b'] = 24;
        extent['c'] = 72;
        extent['d'] = 72;
        extent['e'] = 72;
    }
    else if(equation_num == 5) {
        extent['a'] = 72;
        extent['b'] = 72;
        extent['c'] = 24;
        extent['d'] = 72;
        extent['e'] = 72;
    }
    else if(equation_num == 6) {
        extent['a'] = 48;
        extent['b'] = 32;
        extent['c'] = 24;
        extent['d'] = 32;
        extent['e'] = 48;
        extent['f'] = 32;
    }
    else if(equation_num == 7) {
        extent['a'] = 48;
        extent['b'] = 32;
        extent['c'] = 32;
        extent['d'] = 24;
        extent['e'] = 48;
        extent['f'] = 48;
    }
    else if(equation_num == 8) {
        extent['a'] = 48;
        extent['b'] = 24;
        extent['c'] = 32;
        extent['d'] = 32;
        extent['e'] = 48;
        extent['f'] = 32;
    }
    else if(equation_num == 9) {
        extent['a'] = 72;
        extent['b'] = 72;
        extent['c'] = 72;
        extent['d'] = 72;
        extent['e'] = 72;
    }
    else if(equation_num == 10) {
        extent['a'] = 72;
        extent['b'] = 72;
        extent['c'] = 72;
        extent['d'] = 72;
        extent['e'] = 72;
    }
    else if(equation_num == 11) {
        extent['a'] = 72;
        extent['b'] = 72;
        extent['c'] = 72;
        extent['d'] = 72;
        extent['e'] = 72;
    }
    else if(equation_num == 12) {
        extent['a'] = 5136;
        extent['b'] = 5120;
        extent['c'] = 5136;
    }
    else if(equation_num == 13) {
        extent['a'] = 312;
        extent['b'] = 296;
        extent['c'] = 296;
        extent['d'] = 312;
    }
    else if(equation_num == 14) {
        extent['a'] = 312;
        extent['b'] = 296;
        extent['c'] = 312;
        extent['d'] = 312;
    }
    else if(equation_num == 15) {
        extent['a'] = 312;
        extent['b'] = 296;
        extent['c'] = 296;
        extent['d'] = 312;
    }
    else if(equation_num == 16) {
        extent['a'] = 312;
        extent['b'] = 312;
        extent['c'] = 296;
        extent['d'] = 296;
    }
    else if(equation_num == 17) {
        extent['a'] = 312;
        extent['b'] = 312;
        extent['c'] = 296;
        extent['d'] = 296;
    }
    else if(equation_num == 18) {
        extent['a'] = 312;
        extent['b'] = 296;
        extent['c'] = 296;
        extent['d'] = 312;
    }
    else if(equation_num == 19) {
        extent['a'] = 72;
        extent['b'] = 72;
        extent['c'] = 72;
        extent['d'] = 72;
        extent['e'] = 72;
    }
    else if((equation_num >= 20 && equation_num <= 30)) {
        extent['a'] = 72;
        extent['b'] = 72;
        extent['c'] = 72;
        extent['d'] = 72;
        extent['e'] = 72;
        extent['f'] = 72;
    }
    else if((equation_num >= 31 && equation_num <= 36) || (equation_num >= 43 && equation_num <= 48)) {
		extent['a'] = 24;
        extent['b'] = 16;
        extent['c'] = 16;
        extent['d'] = 24;
        extent['e'] = 16;
        extent['f'] = 16;
        extent['g'] = 24;
	}
	else if((equation_num >= 37 && equation_num <= 42)){
		extent['a'] = 24;
        extent['b'] = 16;
        extent['c'] = 16;
        extent['d'] = 16;
        extent['e'] = 24;
        extent['f'] = 16;
        extent['g'] = 24;
	}

    switch(equation_num) {
        case 1 :
            extentC.clear();
            for(auto mode : {'a','b','c'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'b','d','a'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'d','c'})
                extentB.push_back(extent[mode]);
            break;
        case 2 :
            extentC.clear();
            for(auto mode : {'a','b','c'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'d','c','a'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'b','d'})
                extentB.push_back(extent[mode]);
            break;
        case 3 :
            extentC.clear();
            for(auto mode : {'a','b','c','d'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'d','b','e','a'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'e','c'})
                extentB.push_back(extent[mode]);
            break;
        case 4 :
            extentC.clear();
            for(auto mode : {'a','b','c','d'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'d','e','c','a'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'b','e'})
                extentB.push_back(extent[mode]);
            break;
        case 5 :
            extentC.clear();
            for(auto mode : {'a','b','c','d'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'e','b','a','d'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'c','e'})
                extentB.push_back(extent[mode]);
            break;
        case 6 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'e','f','b','a','d'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'c','f'})
                extentB.push_back(extent[mode]);
            break;
        case 7 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'e','c','b','f','a'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'f','d'})
                extentB.push_back(extent[mode]);
            break;
        case 8 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'e','f','c','a','d'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'b','f'})
                extentB.push_back(extent[mode]);
            break;
        case 9 :
            extentC.clear();
            for(auto mode : {'a','b','c','d'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'e','a'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'e','b','c','d'})
                extentB.push_back(extent[mode]);
            break;
        case 10 :
            extentC.clear();
            for(auto mode : {'a','b','c','d'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'e','b'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'a','e','c','d'})
                extentB.push_back(extent[mode]);
            break;
        case 11 :
            extentC.clear();
            for(auto mode : {'a','b','c','d'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'e','c'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'a','b','e','d'})
                extentB.push_back(extent[mode]);
            break;
        case 12 :
            extentC.clear();
            for(auto mode : {'a','b'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'a','c'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'c','b'})
                extentB.push_back(extent[mode]);
            break;
        case 13 :
            extentC.clear();
            for(auto mode : {'a','b'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'a','c','d'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'d','b','c'})
                extentB.push_back(extent[mode]);
            break;
        case 14 :
            extentC.clear();
            for(auto mode : {'a','b'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'c','a','d'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'d','c','b'})
                extentB.push_back(extent[mode]);
            break;
        case 15 :
            extentC.clear();
            for(auto mode : {'a','b','c'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'a','c','d'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'d','b'})
                extentB.push_back(extent[mode]);
            break;
        case 16 :
            extentC.clear();
            for(auto mode : {'a','b','c'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'a','d'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'b','d','c'})
                extentB.push_back(extent[mode]);
            break;
        case 17 :
            extentC.clear();
            for(auto mode : {'a','b','c'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'a','d','c'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'b','d'})
                extentB.push_back(extent[mode]);
            break;
        case 18 :
            extentC.clear();
            for(auto mode : {'a','b','c'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'a','d','c'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'d','b'})
                extentB.push_back(extent[mode]);
            break;
        case 19 :
            extentC.clear();
            for(auto mode : {'a','b','c'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'a','d','e','c'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'e','b','d'})
                extentB.push_back(extent[mode]);
            break;
        case 20 :
            extentC.clear();
            for(auto mode : {'a','b','c','d'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'a','e','b','f'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'d','f','c','e'})
                extentB.push_back(extent[mode]);
            break;
        case 21 :
            extentC.clear();
            for(auto mode : {'a','b','c','d'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'a','e','b','f'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'f','d','e','c'})
                extentB.push_back(extent[mode]);
            break;
        case 22 :
            extentC.clear();
            for(auto mode : {'a','b','c','d'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'a','e','c','f'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'b','f','d','e'})
                extentB.push_back(extent[mode]);
            break;
        case 23 :
            extentC.clear();
            for(auto mode : {'a','b','c','d'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'a','e','c','f'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'f','b','e','d'})
                extentB.push_back(extent[mode]);
            break;
        case 24 :
            extentC.clear();
            for(auto mode : {'a','b','c','d'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'a','e','d','f'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'b','f','c','e'})
                extentB.push_back(extent[mode]);
            break;
        case 25 :
            extentC.clear();
            for(auto mode : {'a','b','c','d'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'a','e','d','f'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'f','b','e','c'})
                extentB.push_back(extent[mode]);
            break;
        case 26 :
            extentC.clear();
            for(auto mode : {'a','b','c','d'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'a','e','f','b'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'f','d','c','e'})
                extentB.push_back(extent[mode]);
            break;
        case 27 :
            extentC.clear();
            for(auto mode : {'a','b','c','d'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'a','e','f','c'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'f','b','e','d'})
                extentB.push_back(extent[mode]);
            break;
        case 28 :
            extentC.clear();
            for(auto mode : {'a','b','c','d'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'e','a','f','b'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'f','d','e','c'})
                extentB.push_back(extent[mode]);
            break;
        case 29 :
            extentC.clear();
            for(auto mode : {'a','b','c','d'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'e','a','f','c'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'b','f','d','e'})
                extentB.push_back(extent[mode]);
            break;
        case 30 :
            extentC.clear();
            for(auto mode : {'a','b','c','d'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'e','a','f','d'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'f','b','e','c'})
                extentB.push_back(extent[mode]);
            break;
        case 31 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e','f'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'d','e','g','a'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'g','f','b','c'})
                extentB.push_back(extent[mode]);
            break;
        case 32 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e','f'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'d','e','g','b'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'g','f','a','c'})
                extentB.push_back(extent[mode]);
            break;
        case 33 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e','f'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'d','e','g','c'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'g','f','a','b'})
                extentB.push_back(extent[mode]);
            break;
        case 34 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e','f'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'d','f','g','a'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'g','e','b','c'})
                extentB.push_back(extent[mode]);
            break;
        case 35 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e','f'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'d','f','g','b'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'g','e','a','c'})
                extentB.push_back(extent[mode]);
            break;
        case 36 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e','f'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'d','f','g','c'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'g','e','a','b'})
                extentB.push_back(extent[mode]);
            break;
        case 37 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e','f'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'e','f','g','a'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'g','d','b','c'})
                extentB.push_back(extent[mode]);
            break;
        case 38 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e','f'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'e','f','g','b'})
                extentA.push_back(extent[mode]);
                
            extentB.clear();
            for(auto mode : {'g','d','a','c'})
                extentB.push_back(extent[mode]);
            break;
        case 39 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e','f'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'e','f','g','c'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'g','d','a','b'})
                extentB.push_back(extent[mode]);
            break;
        case 40 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e','f'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'g','d','a','b'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'e','f','g','c'})
                extentB.push_back(extent[mode]);
            break;
        case 41 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e','f'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'g','d','a','c'})
                extentA.push_back(extent[mode]);   
            extentB.clear();
            for(auto mode : {'e','f','g','b'})
                extentB.push_back(extent[mode]);
            break;
        case 42 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e','f'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'g','d','b','c'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'e','f','g','a'})
                extentB.push_back(extent[mode]);
            break;
        case 43 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e','f'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'g','e','a','b'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'d','f','g','c'})
                extentB.push_back(extent[mode]);
            break;
        case 44 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e','f'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'g','e','a','c'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'d','f','g','b'})
                extentB.push_back(extent[mode]);
            break;
        case 45 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e','f'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'g','e','b','c'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'d','f','g','a'})
                extentB.push_back(extent[mode]);
            break;
        case 46 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e','f'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'g','f','a','b'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'d','e','g','c'})
                extentB.push_back(extent[mode]);
            break;
        case 47 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e','f'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'g','f','a','c'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'d','e','g','b'})
                extentB.push_back(extent[mode]);
            break;
        case 48 :
            extentC.clear();
            for(auto mode : {'a','b','c','d','e','f'})
                extentC.push_back(extent[mode]);
            extentA.clear();
            for(auto mode : {'g','f','b','c'})
                extentA.push_back(extent[mode]);
            extentB.clear();
            for(auto mode : {'d','e','g','a'})
                extentB.push_back(extent[mode]);
            break;
        default :
            printf("Invalid Equation Number\n");
            break;
    }

    return;
}

double cal_gflops(int equation_num, float elapsed_time, std::unordered_map<int, int64_t> extent) {
    double operations = 2.0;
    for (const auto& [key, val] : extent)
        operations *= val;
    
    double gflops  = operations / 1e9;
    double GFLOPS  = gflops / ((double)elapsed_time / 1000);

    return GFLOPS;
}

std::map<char, int64_t> parse_extents_from_file(int equation_num, int variant) {
    std::string filename = "./model/gen_input/rand_problems_new2/tccg_" + std::to_string(equation_num) + "/problem_" + std::to_string(variant) + ".in";
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        exit(1);
    }

    std::map<char, int64_t> extents;
    std::string line;

    while (std::getline(file, line)) {
        // "[a,9,b,8,c,9]" 형태에서 변수-크기 쌍 추출
        std::regex bracket_pattern(R"(\[([^\]]+)\])");
        std::smatch bracket_match;
        std::string search_line = line;

        while (std::regex_search(search_line, bracket_match, bracket_pattern)) {
            std::string inside = bracket_match[1].str(); // "a,9,b,8,c,9"
            std::stringstream ss(inside);
            std::string token;
            std::vector<std::string> tokens;

            while (std::getline(ss, token, ',')) {
                tokens.push_back(token);
            }

            // 홀수 인덱스: 변수명, 짝수 인덱스: 크기
            for (size_t i = 0; i + 1 < tokens.size(); i += 2) {
                std::string var = tokens[i];
                std::string size_str = tokens[i + 1];

                // 변수가 단일 알파벳이고 크기가 숫자인 경우
                if (var.size() == 1 && std::isalpha(var[0]) && 
                    std::all_of(size_str.begin(), size_str.end(), ::isdigit)) {
                    char key = var[0];
                    int64_t size = std::stoll(size_str);
                    extents[key] = size; // map이므로 중복 시 덮어씀 (일관성 가정)
                }
            }

            search_line = bracket_match.suffix().str();
        }

        // "sum(c,312,d,312)" 형태 처리 - 변수 여러 개 지원
        std::regex sum_pattern(R"(sum\(([^)]+)\))");
        std::smatch sum_match;
        search_line = line;

        while (std::regex_search(search_line, sum_match, sum_pattern)) {
            std::string inside = sum_match[1].str(); // "c,312,d,312"
            std::stringstream ss(inside);
            std::string token;
            std::vector<std::string> tokens;

            while (std::getline(ss, token, ',')) {
                tokens.push_back(token);
            }

            // 변수-크기 쌍으로 파싱
            for (size_t i = 0; i + 1 < tokens.size(); i += 2) {
                std::string var = tokens[i];
                std::string size_str = tokens[i + 1];

                if (var.size() == 1 && std::isalpha(var[0]) &&
                    std::all_of(size_str.begin(), size_str.end(), ::isdigit)) {
                    extents[var[0]] = std::stoll(size_str);
                }
            }

            search_line = sum_match.suffix().str();
        }
    }

    return extents;
}

std::vector<std::string> extents_to_argv(const std::map<char, int64_t>& extents) {
    std::vector<std::string> args;
    for (const auto& [key, value] : extents) { // map은 이미 알파벳 순 정렬
        args.push_back(std::to_string(value));
    }
    return args;
}
