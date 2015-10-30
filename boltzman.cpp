#include <iostream>
#include <string>
#include <random>
#include <functional>
#include <cmath>
#include <algorithm>

using namespace std;

template<int nvisible,int nhidden, typename random_engine_t>
class bm {
public:
	random_engine_t &engine;
	void update(bool *start,int size){
		for(bool *p=start;p<start+size;p++){
			double E_nochange = E();
			*p = !*p;
			double E_change = E();
			double b_change = exp(-E_change/T);
			double b_nochange = exp(-E_nochange/T);
			double p_change = b_change / ( b_change + b_nochange );
			double r = uniform_real_distribution<double>(0,1)(engine);
			if(r>p_change)
				*p = !*p;
		}
	}

	bm(random_engine_t &engine):engine(engine){
		uniform_real_distribution<double> dist(-1,1);
		auto random = bind(dist,ref(engine));
		for(int i=0;i<nvisible+nhidden;i++){
			w[i][i] = 0;
			for(int j=i+1;j<nvisible+nhidden;j++)
				w[i][j] = w[j][i] = random();
		}
		for(double &i: bvisible)
			i = random();
		for(double &i: bhidden)
			i = random();
		for(bool &i:visible)
			i = random()>0;
		for(bool &i:hidden)
			i = random()>0;
	}

	bool visible[nvisible];
	bool hidden[nhidden];
	double w[nvisible+nhidden][nvisible+nhidden]; //visible goes first, then hidden
	double bvisible[nvisible];
	double bhidden[nhidden];
	double T = 1;

	double E() {
		double ret = 0;
		for(int i=0;i<nvisible;i++){
			ret -= bvisible[i]*visible[i];
			for(int j=i+1;j<nvisible;j++)
				ret -= w[i][j]*visible[i]*visible[j];
			for(int j=0;j<nhidden;j++)
				ret -= w[i][nvisible+j]*visible[i]*hidden[j];
		}
		for(int i=0;i<nhidden;i++){
			ret -= bhidden[i]*hidden[i];
			for(int j=i+1;j<nhidden;j++)
				ret -= w[nvisible+i][nvisible+j]*hidden[i]*hidden[j];
		}
		return ret;
	}

	void update_visibles(){
		update(&visible[0],nvisible);
	}

	void update_hiddens(){
		update(&hidden[0],nhidden);
	}

	virtual double train1(const double rate, const int trainsize, bool trainingdata[][nvisible], int samplesize){
		double dw[nvisible+nhidden][nvisible+nhidden] = { 0 };
		double dbvis[nvisible] = { 0 };
		double dbhid[nhidden];
		for(double &i:dbhid) i=0;
		
		///////////////// phase + //////////////////
		// calculate <ViVj>
		for(int i=0;i<nvisible;i++){
			for(int j=i+1;j<nvisible;j++){
				double dwij = 0;
				for(int k=0;k<trainsize;k++)
					dwij += ((double)trainingdata[k][i])*((double)trainingdata[k][j]);
				dw[i][j] = dw[j][i] = dwij/trainsize;
			}
		}
		// calculate <Vi>
		for(int i=0;i<nvisible;i++){
			double dbi = 0;
			for(int k=0;k<trainsize;k++)
				dbi += (double)trainingdata[k][i];
			dbvis[i] = dbi/trainsize;
		}

		// calculate <ViHj> <Hj> <HiHj>
		double vihj[nvisible][nhidden];
		for(int i=0;i<nvisible;i++)
			for(int j=0;j<nhidden;j++)
				vihj[i][j] = 0;
		double hj[nhidden];
		for(double &i:hj) { i = 0; }
		double hihj[nhidden][nhidden];
		for(int i=0;i<nhidden;i++)
			for(int j=0;j<nhidden;j++)
				hihj[i][j] = 0;
		for(int i=0;i<trainsize;i++){
			// clamp visible
			copy(&trainingdata[i][0],&trainingdata[i][nvisible],&visible[0]);
			// update hidden and wait until thermal eq
			for(int j=0;j<samplesize;j++)
				update_hiddens();
			// sample hidden with visible clamped
			bool ghidden[samplesize][nhidden];
			for(int j=0;j<samplesize;j++){
				update_hiddens();
				copy(&hidden[0],&hidden[nhidden],&ghidden[j][0]);
			}
			// calculate <HiHj> for this train data
			for(int j=0;j<nhidden;j++){
				for(int k=0;k<nhidden;k++){
					double hihjjk = 0;
					for(int l=0;l<samplesize;l++)
						hihjjk += ((double)ghidden[l][j])*((double)ghidden[l][k]);
					hihj[j][k] += hihjjk/samplesize;
				}
			}
			// calculate <Hj> and <ViHj> for this train data
			for(int j=0;j<nhidden;j++){
				double hjj = 0;
				for(int l=0;l<samplesize;l++)
					hjj += ((double)ghidden[l][j]);
				hj[j] += hjj/samplesize;
				for(int k=0;k<nvisible;k++)
					vihj[k][j] += trainingdata[i][k]*hj[j];
			}
		}
		for(int i=0;i<nhidden;i++){
			dbhid[i] = hj[i]/trainsize;
			for(int j=i+1;j<nvisible;j++)
				dw[nvisible+i][j] = dw[j][nvisible+i] = vihj[j][i]/trainsize;
		}
		for(int i=0;i<nhidden;i++)
			for(int j=i+1;j<nhidden;j++)
				dw[nvisible+i][nvisible+j] = dw[nvisible+j][nvisible+i] = hihj[i][j]/trainsize;
		
		///////////////// phase - //////////////////
		// wait until thermal eq
		for(int i=0;i<samplesize;i++){
			update_hiddens();
			update_visibles();
		}
		{ // use block to keep the stack small
		bool vishid[samplesize][nvisible+nhidden];
		// sample visible and hidden
		for(int i=0;i<samplesize;i++){
			update_hiddens();
			update_visibles();
			copy(&visible[0],&visible[nvisible],&vishid[i][0]);
			copy(&hidden[0],&hidden[nhidden],&vishid[i][nvisible]);
		}
		// calculate <Vi> for model
		for(int i=0;i<nvisible;i++){
			double si = 0;
			for(int j=0;j<samplesize;j++)
				si += (double)vishid[j][i];
			dbvis[i] -= si/samplesize;
		}
		// calculate <Hi> for model
		for(int i=0;i<nhidden;i++){
			double si = 0;
			for(int j=0;j<samplesize;j++)
				si += (double)vishid[j][nvisible+i];
			dbhid[i] -= si/samplesize;
		}
		// calculate <SiSj> for model
		for(int i=0;i<nvisible+nhidden;i++){
			for(int j=i+1;j<nvisible+nhidden;j++){
				double dwij = 0;
				for(int k=0;k<samplesize;k++)
					dwij += ((double)vishid[k][i])*((double)vishid[k][j]);
				dw[i][j] -= dwij/samplesize;
				dw[j][i] -= dwij/samplesize;
			}
		}
		/* end block */ }

		///////////////// update //////////////////
		for(int i=0;i<nvisible;i++)
			bvisible[i] += rate*dbvis[i];
		for(int i=0;i<nhidden;i++)
			bhidden[i] += rate*dbhid[i];
		for(int i=0;i<nvisible+nhidden;i++)
			for(int j=0;j<nvisible+nhidden;j++)
				w[i][j] += rate*dw[i][j];
		////////////// calculate err ///////////////
		double error = 0;
		for(int i=0;i<nvisible;i++)
			error = max(error,abs(dbvis[i]));
		for(int i=0;i<nhidden;i++)
			error = max(error,abs(dbhid[i]));
		for(int i=0;i<nvisible+nhidden;i++)
			for(int j=0;j<nvisible+nhidden;j++)
				error = max(error,abs(dw[i][j]));
		return error;
	}

	void train(const double rate, const int trainsize, bool trainingdata[][nvisible], int samplesize,
	           function<void (int,double)> callback = [](int,double){} , double tol = 1e-6){
		double err;
		int count = 0;
		while( (err=train1(rate,trainsize,trainingdata,samplesize)) > tol )
			callback(count++,err);
		callback(count++,err);
	}
};

template<int nvisible,int nhidden, typename random_engine_t>
class rbm:public bm<nvisible,nhidden,random_engine_t>{
public:
	rbm(random_engine_t &engine):bm<nvisible,nhidden,random_engine_t>(engine){
		// remove coupling within layer
		for(int i=0;i<nvisible;i++)
			for(int j=0;j<nvisible;j++)
				this->w[i][j] = 0;
		for(int i=0;i<nhidden;i++)
			for(int j=0;j<nhidden;j++)
				this->w[nvisible+i][nvisible+j] = 0;
	}
	
	virtual double train1(const double rate, const int trainsize, bool trainingdata[][nvisible],int cd){
		double dw[nvisible][nhidden] = { 0 };
		double dbvis[nvisible] = { 0 };
		double dbhid[nhidden] = { 0 };

		///////////////// phase + //////////////////
		// calculate <Vi>
		for(int i=0;i<nvisible;i++){
			double dbi = 0;
			for(int k=0;k<trainsize;k++)
				dbi += (double)trainingdata[k][i];
			dbvis[i] = dbi/trainsize;
		}
		// calculate <Hi> <VjHi>
		double dbh[nhidden] = { 0 };
		double dvh[nvisible][nhidden] = { 0 };
		for(int k=0;k<trainsize;k++){
			copy(&trainingdata[k][0],&trainingdata[k][nvisible],&this->visible[0]);
			double E_nochange = this->E();
			double b_nochange = exp(-E_nochange/this->T);
			for(int i=0;i<nhidden;i++){
				this->hidden[i] = !this->hidden[i] ;
				double E_change = this->E();
				this->hidden[i] = !this->hidden[i] ;
				double b_change = exp(-E_change/this->T);
				double hi = (b_change*((double)!this->hidden[i])+b_nochange*((double)this->hidden[i]))/(b_change+b_nochange);
				dbh[i] += hi;
				for(int j=0;j<nvisible;j++)
					dvh[j][i] += ((double)this->visible[j])*hi;
			}
		}
		for(int i=0;i<nhidden;i++){
			dbhid[i] = dbh[i]/trainsize;
			for(int j=0;j<nvisible;j++)
				dw[j][i] += dvh[j][i]/trainsize;
		}

		///////////////// phase - //////////////////
		double dbv[nvisible] = { 0 };
		for(double &i:dbh) i=0;
		fill(&dvh[0][0],&dvh[0][0]+nvisible*nhidden,0.0);
		for(int k=0;k<trainsize;k++){
			copy(&trainingdata[k][0],&trainingdata[k][nvisible],&this->visible[0]);
			this->update_hiddens();
			for(int i=0;i<cd;i++){
				this->update_visibles();
				this->update_hiddens();
			}
			// calculate <Vi>
			for(int i=0;i<nvisible;i++)
				dbv[i] += (double)this->visible[i];
			// calculate <Hi>
			for(int i=0;i<nhidden;i++)
				dbh[i] += (double)this->hidden[i];
			// calculate <ViHj>
			for(int i=0;i<nvisible;i++)
				for(int j=0;j<nhidden;j++)
					dvh[i][j] += ((double)this->visible[i])*((double)this->hidden[j]);
		}
		for(int i=0;i<nvisible;i++)
			dbvis[i] -= dbv[i]/trainsize;
		for(int i=0;i<nhidden;i++)
			dbhid[i] -= dbh[i]/trainsize;
		for(int i=0;i<nvisible;i++)
			for(int j=0;j<nhidden;j++)
				dw[i][j] -= dvh[i][j]/trainsize;

		///////////////// update //////////////////
		for(int i=0;i<nvisible;i++)
			this->bvisible[i] += rate*dbvis[i];
		for(int i=0;i<nhidden;i++)
			this->bhidden[i] += rate*dbhid[i];
		for(int i=0;i<nvisible;i++){
			for(int j=0;j<nhidden;j++){
				this->w[i][nvisible+j] += rate*dw[i][j];
				this->w[nvisible+j][i] = this->w[i][nvisible+j];
			}
		}

		////////////// calculate err ///////////////
		double error = 0;
		for(int i=0;i<nvisible;i++)
			error = max(error,abs(dbvis[i]));
		for(int i=0;i<nhidden;i++)
			error = max(error,abs(dbhid[i]));
		for(int i=0;i<nvisible;i++)
			for(int j=0;j<nhidden;j++)
				error = max(error,abs(dw[i][j]));
		return error;
	}

};

template <typename generate_t, typename learn_t>
void generate_and_learn(generate_t &g, learn_t &l, double rate, int trainsize=1000, int samplesize=1, int skip=1){
	const int nvisible = sizeof(g.visible)/sizeof(bool);
	const int nhidden1 = sizeof(g.hidden)/sizeof(bool);
	const int nhidden2 = sizeof(l.hidden)/sizeof(bool);
	bool training[trainsize][nvisible];
	// wait for thermal eq
	for(int i=0;i<trainsize;i++){
		g.update_hiddens();
		g.update_visibles();
	}
	// generate training data
	for(int i=0;i<trainsize;i++){
		g.update_hiddens();
		g.update_visibles();
		copy(&g.visible[0],&g.visible[nvisible],&training[i][0]);
	}
	// train l
	l.train(rate,trainsize,training,samplesize,[&](int c, double x){
		if(c%skip!=0) return;
		double err = 0;
		for(int i=0;i<nvisible;i++)
			err = max(err,abs(l.bvisible[i]-g.bvisible[i]));
		if(nhidden1==nhidden2){
			int nhidden = nhidden1;
			for(int i=0;i<nhidden;i++)
				err = max(err,abs(l.bhidden[i]-g.bhidden[i]));
			for(int i=0;i<nvisible+nhidden;i++)
				for(int j=0;j<nvisible+nhidden;j++)
					err = max(err,abs(l.w[i][j]-g.w[i][j]));
		}
		//cout << "--------------------------\n";
		cout << c << '\t' << x << '\t' << err << "\n";
		/*cout << "---\nvis: ";
		for(int i=0;i<nvisible;i++)
			cout << l.visible[i] << "\t";
		cout << "\nhid: ";
		for(int i=0;i<nhidden2;i++)
			cout << l.hidden[i] << "\t";
		cout << "\nenergy: ";
		cout << l.E() << "\n";
		cout << "---\n";
		for(int i=0;i<nvisible+nhidden2;i++){
			for(int j=0;j<nvisible+nhidden2;j++)
				cout << l.w[i][j] << "\t";
			cout << "\n";
		}
		cout << "---\nbvis:";
		for(int i=0;i<nvisible;i++)
			cout << l.bvisible[i] << "\t";
		cout << "\nbhid: ";
		for(int i=0;i<nhidden2;i++)
			cout << l.bhidden[i] << "\t";
		cout << "\n";*/
	});
}

int main(){
	// Seed with a real random value, if available
	random_device rd;
	default_random_engine rengine(rd());
	//////////////// Model 0 /////////////////
	rbm<2,2,default_random_engine> source(rengine);
	rbm<2,2,default_random_engine> learn(rengine);
	generate_and_learn(source,learn, 0.001, 5000, 5, 100);
	/*
	bm<3,2,default_random_engine> m(rengine);
	int stat[32] = { 0 };
	double e[32] = { 0 };
	for(int i=0;i<10000000;i++){
		m.update_hiddens();
		m.update_visibles();
	}
	for(int i=0;i<10000000;i++){
		m.update_hiddens();
		m.update_visibles();
		int idx = 0;
		idx = ((int)m.hidden[0])*16+((int)m.hidden[1])*8+((int)m.visible[0])*4+((int)m.visible[1])*2+((int)m.visible[2]);
		stat[idx]++;
		e[idx] = m.E();
	}
	for(int i=0;i<32;i++)
		cout << stat[i] << "\t" << e[i] << endl;*/
}
