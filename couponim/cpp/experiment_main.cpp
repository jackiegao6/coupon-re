#include "coupon_core.hpp"

void run_forward(Graph& g,istream& input,ostream& out) {
    auto batches=read_one<uint32_t>(input);PythonMT rng(input);Walker walker(g);
    for(uint32_t v=0;v<g.n;++v) {
        vector<int64_t> outcomes;
        for(uint32_t b=0;b<batches;++b) {auto u=walker.walk(v,rng);if(u>=0)outcomes.push_back(int64_t(b)*g.n+u);}
        write_one(out,uint32_t(outcomes.size()));out.write(reinterpret_cast<char*>(outcomes.data()),outcomes.size()*sizeof(int64_t));
    }
}

void run_evaluation(Graph& g,istream& input,ostream& out) {
    uint32_t size=read_one<uint32_t>(input),trials=read_one<uint32_t>(input);if(!trials)throw runtime_error("trials must be positive");Walker walker(g);vector<uint64_t> consumed(g.n);uint64_t epoch=0;
    out<<setprecision(17)<<'[';
    for(uint32_t i=0;i<size;++i) {
        auto start=Clock::now();uint32_t k=read_one<uint32_t>(input);auto seeds=read_vec<int32_t>(input,k);PythonMT rng(input);
        double mean=0,m2=0,redeemed_mean=0;
        for(uint32_t t=1;t<=trials;++t) {
            ++epoch;uint32_t users=0,redeemed=0;
            for(auto s:seeds) {auto u=walker.walk(s,rng);if(u>=0){++redeemed;if(consumed[u]!=epoch){consumed[u]=epoch;++users;}}}
            double delta=users-mean;mean+=delta/t;m2+=delta*(users-mean);redeemed_mean+=(redeemed-redeemed_mean)/t;
        }
        double variance=trials>1?m2/(trials-1):0,ci=1.96*sqrt(variance/trials);
        if(i)out<<',';out<<"{\"spread\":"<<mean<<",\"within_run_ci95_halfwidth\":"<<ci<<",\"evaluation_variance\":"<<variance
                       <<",\"consumed_coupons\":"<<redeemed_mean<<",\"evaluation_seconds\":"<<elapsed(start)<<'}';
    }
    out<<']';
}

int main(int argc,char** argv) {
    try {
        if(argc!=5)throw runtime_error("usage: kernel mode graph.bin input.bin output");
        ifstream graphfile(argv[2],ios::binary),input(argv[3],ios::binary);ofstream output(argv[4],ios::binary);Graph graph(graphfile);
        string mode=argv[1];
        if(mode=="forward")run_forward(graph,input,output);
        else if(mode=="evaluate")run_evaluation(graph,input,output);
        else throw runtime_error("unknown mode");
        if(!output)throw runtime_error("write failed");
    } catch(const exception& e) {cerr<<e.what()<<endl;return 1;}
    return 0;
}
