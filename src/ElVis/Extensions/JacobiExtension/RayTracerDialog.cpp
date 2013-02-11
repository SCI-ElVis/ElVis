//////////////////////////////////////////////////////////////////////////////////
////
////  File: hoRayTracerDialog.h
////
////  For more information, please see: http://www.nektar.info/
////
////  The MIT License
////
////  Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
////  Department of Aeronautics, Imperial College London (UK), and Scientific
////  Computing and Imaging Institute, University of Utah (USA).
////
////  License for the specific language governing rights and limitations under
////  Permission is hereby granted, free of charge, to any person obtaining a
////  copy of this software and associated documentation files (the "Software"),
////  to deal in the Software without restriction, including without limitation
////  the rights to use, copy, modify, merge, publish, distribute, sublicense,
////  and/or sell copies of the Software, and to permit persons to whom the
////  Software is furnished to do so, subject to the following conditions:
////
////  The above copyright notice and this permission notice shall be included
////  in all copies or substantial portions of the Software.
////
////  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
////  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
////  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
////  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
////  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
////  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
////  DEALINGS IN THE SOFTWARE.
////
////  Description:
////
//////////////////////////////////////////////////////////////////////////////////
//
//#include <ElVis/Extensions/JacobiExtension/RayTracerDialog.h>
//#include <ElVis/Extensions/JacobiExtension/qtRayTracerDialog.h>
//#include <ElVis/Extensions/JacobiExtension/Plugin.h>
//
//#include <boost/lexical_cast.hpp>
//#include <iostream>
//
//using std::cout;
//using std::endl;
//
//namespace JacobiExtension
//{
//	class RayTracerDialog::RayTracerDialogImpl
//	{
//		public:
//			RayTracerDialogImpl(HighOrderIsosurfacePlugin* thePlugin) :
//				dialog(),
//				plugin(thePlugin)
//			{
//			}
//
//			Ui::RayTracerDialog dialog;
//			HighOrderIsosurfacePlugin* plugin;
//	};
//				
//	RayTracerDialog::RayTracerDialog(HighOrderIsosurfacePlugin* plugin, QWidget* parent, Qt::WFlags f) :
//		QDialog(parent, f),
//		m_impl(new RayTracerDialogImpl(plugin))
//	{
//		m_impl->dialog.setupUi(this);
//		QString minValue("0.0");
//		QString maxValue("0.0");
//
//		m_impl->dialog.minScalarValueInput->setText(minValue);
//		m_impl->dialog.maxScalarValueInput->setText(maxValue);
//		m_impl->dialog.scalarValueInput->setText(minValue);
//
//		connect(m_impl->dialog.rayTraceButton, SIGNAL(clicked()), m_impl->plugin, SLOT(rayTraceScene()));
//	}
//
//	RayTracerDialog::~RayTracerDialog()
//	{
//	}
//
//	void RayTracerDialog::setMinScalarValue(double newVal)
//	{
//		std::string asString = boost::lexical_cast<std::string>(newVal);
//		QString asQString(asString.c_str());
//		m_impl->dialog.minScalarValueInput->setText(asQString);
//	}
//
//	void RayTracerDialog::setMaxScalarValue(double newVal)
//	{
//		std::string asString = boost::lexical_cast<std::string>(newVal);
//		QString asQString(asString.c_str());
//		m_impl->dialog.maxScalarValueInput->setText(asQString);
//	}
//
//	double RayTracerDialog::getScalarValueToRayTrace() const
//	{
//		std::string val = m_impl->dialog.scalarValueInput->text().toStdString();
//		try
//		{
//			return boost::lexical_cast<double>(val);
//		}
//		catch(...)
//		{
//			cout << "Can't convert string." << endl;
//			return 0.0;
//		}
//	}
//
//	QProgressBar* RayTracerDialog::getProgressBar()
//	{
//		return m_impl->dialog.rayTracerProgress;
//	}
//}
