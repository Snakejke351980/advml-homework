#include <iostream>
#include <chrono>
#include <string>
#include <random>
#include <functional>
#include <cmath>
#include <algorithm>
#include <cmath>

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
	
	int idx() {
		int p = 1;
		int ret = 0;
		for(int i=0;i<nvisible;i++){
			ret += p*((int)visible[i]);
			p *= 2;
		}
		return ret;
	}

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

	virtual void train(const double rate, const int trainsize, bool trainingdata[][nvisible], int samplesize){
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
			/* don't wait for "thermal eq"
			// update hidden and wait until thermal eq
			for(int j=0;j<samplesize;j++)
				update_hiddens();
			*/
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
				hjj /= samplesize;
				hj[j] += hjj;
				for(int k=0;k<nvisible;k++)
					vihj[k][j] += ((double)trainingdata[i][k])*((double)hjj);
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
		/* don't wait for "thermal eq"
		// wait until thermal eq
		for(int i=0;i<samplesize;i++){
			update_hiddens();
			update_visibles();
		}
		*/
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
		
		///////////////// debug ///////////////////
		/*
		cout << "-----------------------\n";
		for(int i=0;i<nvisible;i++)
			cout << dbvis[i] << "\t";
		cout << "\n";
		for(int i=0;i<nhidden;i++)
			cout << dbhid[i] << "\t";
		cout << "\n";
		for(int i=0;i<nvisible+nhidden;i++){
			for(int j=0;j<nvisible+nhidden;j++){
				cout << dw[i][j] << "\t";
			}
			cout << "\n";
		}
		cout << "-----------------------\n";*/
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
	
	virtual void train(const double rate, const int trainsize, bool trainingdata[][nvisible],int cd){
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
	}

};

template <typename generate_t, typename learn_t>
void generate_and_learn(generate_t &g, learn_t &l, double rate, int trainsize, int max_epochs, int szcd, int skip){
	const int nvisible = sizeof(g.visible)/sizeof(bool);
	const int nhidden1 = sizeof(g.hidden)/sizeof(bool);
	const int nhidden2 = sizeof(l.hidden)/sizeof(bool);
	const int ns = pow(2,nvisible);
	bool training[trainsize][nvisible];
	int stat_training[ns];
	int stat_learned[ns];
	fill(&stat_training[0],&stat_training[ns],0);
	fill(&stat_learned[0], &stat_learned[ns],0);
	/* don't wait for "thermal eq"
	// wait for thermal eq
	for(int i=0;i<trainsize;i++){
		g.update_hiddens();
		g.update_visibles();
	}
	*/
	// generate training data
	for(int i=0;i<trainsize;i++){
		g.update_hiddens();
		g.update_visibles();
		copy(&g.visible[0],&g.visible[nvisible],&training[i][0]);
		stat_training[g.idx()]++;
	}
	// train l
	chrono::duration<double> elapsed_seconds(0);
	for(int c=0;c<max_epochs;c++){
		
		auto start = chrono::steady_clock::now();
		l.train(rate,trainsize,training,szcd);
		auto end = chrono::steady_clock::now();
		
		elapsed_seconds += chrono::duration_cast<chrono::duration<double>>(end-start);
		
		if(c%skip!=0) continue;
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
		cout << c << '\t' << err << "\n";
	}
	// output runing time
	cout << "runing time: " << elapsed_seconds.count() << " seconds\n";
	// statistics of learned
	for(int i=0;i<trainsize;i++){
		l.update_hiddens();
		l.update_visibles();
	}
	for(int i=0;i<trainsize;i++){
		l.update_hiddens();
		l.update_visibles();
		stat_learned[l.idx()]++;
	}
	// print statistics
	cout << "Statistics:\ntraining\tlearned\n";
	for(int i=0;i<ns;i++)
		cout << stat_training[i] << "\t" << stat_learned[i] << "\n";
}

int main(){
	// Seed with a real random value, if available
	random_device rd;
	default_random_engine rengine(rd());
	//////////////// Model 0 /////////////////
	rbm<3,3,default_random_engine> source(rengine);
	rbm<3,4,default_random_engine> learn(rengine);
	generate_and_learn(source,learn, 0.01,10000, 30000, 1, 100);
}
