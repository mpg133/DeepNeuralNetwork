#pragma once
#include <vector>
#include "Config.h"
#include "RNG.h"
#include "lib/Eigen/Core"

class Layer{


protected:
    //Scalar=Double, Matrix(Double,Dynamic,Dynamic)
    typedef Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar,Eigen::Dynamic,1> Vector;

    const int m_in_size;
    const int m_out_size;


public: 
    //Constructor, inicia el tamaño de la matriz
    Layer(const int in_size, const int out_size):
        m_in_size(in_size),
        m_out_size(out_size){}

    virtual ~Layer();

    int get_in_size() const {return m_in_size;}
    int get_out_size() const {return m_out_size;}

    //virtual function que inicializa el peso de la matriz
    //se le pasa el RNG para generar los pesos tb
    virtual void init(const Scalar&, const Scalar &sigma, RNG& rng) = 0;
    
    virtual void forward(const Matrix& prev_layer_output) = 0;

    virtual const Matrix& output() const = 0;

    //función que actualiza los pesos 
    virtual void backprop(const Matrix& pre_layer_output, const Matrix& next_layer_data)=0;

    virtual const Matrix& backprop_data() const = 0;
    //getters and setters
    virtual std::vector<Scalar>get_paramaeter() const = 0;
    virtual void set_parameters(const std::vector <Scalar>& param) {}
    virtual std::vector<Scalar> get_derivatives() const = 0;
private:


};

Layer::Layer(){

}
Layer::~Layer(){

}