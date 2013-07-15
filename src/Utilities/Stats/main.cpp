///////////////////////////////////////////////////////////////////////////////
//
// The MIT License
//
// Copyright (c) 2006 Scientific Computing and Imaging Institute,
// University of Utah (USA)
//
// License for the specific language governing rights and limitations under
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
///////////////////////////////////////////////////////////////////////////////


#include <iostream>
#include <stdio.h>
#include <math.h>
#include <ElVis/Core/Float.h>
#include <fstream>
#include <string>
#include <ElVis/Core/Cuda.h>
#include <stdlib.h>

class BinaryFile
{
    public:
        BinaryFile(char* name) :
            filePtr(0),
            fileName(name)
        {
        }

        void Open()
        {
            std::cout << "Opening file " << fileName << std::endl;
            filePtr = fopen(fileName, "rb");
            if( !filePtr )
            {
                std::string error = std::string("Can't open file ") + std::string(fileName);
                throw error;
            }

            fread(&width, sizeof(unsigned int), 1, filePtr);
            fread(&height, sizeof(unsigned int), 1, filePtr);
            fread(&numSamples, sizeof(int), 1, filePtr);
            std::cout << "File has size " << width << ", " << height << std::endl;
            std::cout << "Generated with " << numSamples << " samples." << std::endl;
            std::cout << "%%%%%%%%%%%%%%%%%%%%" << std::endl;
            std::cout << numSamples << std::endl;
            std::cout << "%%%%%%%%%%%%%%%%%%%%" << std::endl;
        }

    protected:
        FILE* filePtr;
        unsigned int width;
        unsigned int height;
        int numSamples;

    private:
        char* fileName;
};

class DensityFile : public BinaryFile
{
    public:
        DensityFile(char* name) : BinaryFile(name), data(0) {}
        ~DensityFile()
        {
            delete data;
            data = 0;
        }

        void Read()
        {
            data = new ElVisFloat[width*height];
            fread(data, sizeof(ElVisFloat), width*height, filePtr);
        }

        static void CalculateInfinityError(const DensityFile& lhs, const DensityFile& rhs)
        {
            if( lhs.width != rhs.width ||
                    lhs.height != rhs.height )
            {
                std::string error = std::string("Files are not the same size, can't caluclate infinity error.");
                throw error;
            }

            unsigned int maxRow;
            unsigned int maxCol;
            unsigned int maxIndex;
            ElVisFloat maxError = -1.0;

            for(unsigned int row = 0; row < lhs.height; ++row)
            {
                for(unsigned int col = 0; col < lhs.width; ++col)
                {
                    int index = row*lhs.width + col;
                    ElVisFloat error = fabs(lhs.data[index] - rhs.data[index]);
                    if( error > maxError )
                    {
                        maxError = error;
                        maxRow = row;
                        maxCol = col;
                        maxIndex = index;
                    }
                }
            }

            std::cout.precision(15);
            std::cout << "Max error is at (" << maxRow << ", " << maxCol << " with value " << maxError << std::endl;
            std::cout << "Left Value: " << lhs.data[maxIndex] << std::endl;
            std::cout << "Right Value: " << rhs.data[maxIndex] << std::endl;
        }

    private:
        ElVisFloat* data;
};

std::ostream& operator<<(std::ostream& os, const ElVisFloat3& lhs)
{
    os << "(" << lhs.x << ", " << lhs.y << ", " << lhs.z << ")";
    return os;
}

class ColorFile : public BinaryFile
{
    public:
        ColorFile(char* name) : BinaryFile(name), data(0) {}
        ~ColorFile()
        {
            delete data;
            data = 0;
        }

        void Read()
        {
            data = new ElVisFloat3[width*height];
            fread(data, sizeof(ElVisFloat3), width*height, filePtr);
        }

        static void CalculateErrors(const ColorFile& lhs, const ColorFile& rhs, std::ostream& maxErrorFile, std::ostream& rmseFile)
        {
            if( lhs.width != rhs.width ||
                    lhs.height != rhs.height )
            {
                std::string error = std::string("Files are not the same size, can't caluclate infinity error.");
                throw error;
            }

            uint3 maxRow;
            uint3 maxCol;
            uint3 maxIndex;
            ElVisFloat3 maxError;
            maxError.x = -1.0;
            maxError.y = -1.0;
            maxError.z = -1.0;

            ElVisFloat maxDistance = -1;
            unsigned int distanceRow;
            unsigned int distanceCol;
            unsigned int distanceIndex;

            ElVisFloat mse = 0.0;

            for(unsigned int row = 0; row < lhs.height; ++row)
            {
                for(unsigned int col = 0; col < lhs.width; ++col)
                {
                    int index = row*lhs.width + col;
                    ElVisFloat3 error;
                    error.x = fabs(lhs.data[index].x - rhs.data[index].x);
                    error.y = fabs(lhs.data[index].y - rhs.data[index].y);
                    error.z = fabs(lhs.data[index].z - rhs.data[index].z);

                    if( error.x > maxError.x )
                    {
                        maxError.x = error.x;
                        maxRow.x = row;
                        maxCol.x = col;
                        maxIndex.x = index;
                    }

                    if( error.y > maxError.y )
                    {
                        maxError.y = error.y;
                        maxRow.y = row;
                        maxCol.y = col;
                        maxIndex.y = index;
                    }

                    if( error.z > maxError.z )
                    {
                        maxError.z = error.z;
                        maxRow.z = row;
                        maxCol.z = col;
                        maxIndex.z = index;
                    }

                    ElVisFloat distance = sqrt(error.x*error.x + error.y*error.y + error.z*error.z);
                    if( distance > maxDistance )
                    {
                        maxDistance = distance;
                        distanceIndex = index;
                        distanceRow = row;
                        distanceCol = col;
                    }

                    mse += distance*distance;
                }
            }

            std::cout.precision(15);
            std::cout << "Max red error is at (" << maxRow.x << ", " << maxCol.x << " with value " << maxError.x << std::endl;
            std::cout << "Left Value: " << lhs.data[maxIndex.x].x << std::endl;
            std::cout << "Right Value: " << rhs.data[maxIndex.x].x << std::endl;

            std::cout << "Max green error is at (" << maxRow.y << ", " << maxCol.y << " with value " << maxError.y << std::endl;
            std::cout << "Left Value: " << lhs.data[maxIndex.y].y << std::endl;
            std::cout << "Right Value: " << rhs.data[maxIndex.y].y << std::endl;

            std::cout << "Max blue error is at (" << maxRow.z << ", " << maxCol.z << " with value " << maxError.z << std::endl;
            std::cout << "Left Value: " << lhs.data[maxIndex.z].z << std::endl;
            std::cout << "Right Value: " << rhs.data[maxIndex.z].z << std::endl;

            std::cout << "Max distance error is at (" << distanceRow << ", " << distanceCol << ") with value " << maxDistance << std::endl;
            std::cout << "##############" << std::endl;
            std::cout << maxDistance << std::endl;
            std::cout << "##############" << std::endl;
            std::cout << "Left Value: " << lhs.data[distanceIndex] << std::endl;
            std::cout << "Right Value: " << rhs.data[distanceIndex] << std::endl;

            double rmse = sqrt(mse/(lhs.height*lhs.width));
            std::cout << "RMSE = " << rmse << std::endl;

            maxErrorFile.precision(15);
            rmseFile.precision(15);
            maxErrorFile << "(" << rhs.numSamples << ", " << maxDistance << ")" << std::endl;
            rmseFile << "(" << rhs.numSamples << ", " << rmse << ")" << std::endl;
        }

    private:
        ElVisFloat3* data;
};

int main(int argc, char** argv)
{
    std::cout << "Starting Stats." << std::endl;
    try
    {
        char* file1Name = argv[1];
        char* file2Name = argv[2];
        int type = atoi(argv[3]);

        std::cout << "File 1 = " << file1Name << std::endl;
        std::cout << "File 2 = " << file2Name << std::endl;

        if( type == 0 )
        {
            DensityFile file1(file1Name);
            DensityFile file2(file2Name);

            file1.Open();
            file2.Open();
            file1.Read();
            file2.Read();

            DensityFile::CalculateInfinityError(file1, file2);
        }
        else
        {
            ColorFile file1(file1Name);
            ColorFile file2(file2Name);

            file1.Open();
            file2.Open();
            file1.Read();
            file2.Read();

            std::ofstream maxErrorFile("maxError.txt", std::ios::app);
            std::ofstream rmseErrorFile("rmseError.txt", std::ios::app);
            ColorFile::CalculateErrors(file1, file2, maxErrorFile, rmseErrorFile);
            maxErrorFile.close();
            rmseErrorFile.close();
        }
    }
    catch(std::string& e)
    {
        std::cout << "Error " << e << std::endl;
        return 1;
    }

}

