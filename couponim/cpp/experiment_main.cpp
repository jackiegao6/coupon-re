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

void run_ssr(Graph& g,istream& input,ostream& out) {
    auto start=Clock::now();
    uint32_t theta=read_one<uint32_t>(input),oldk=read_one<uint32_t>(input),kmax=read_one<uint32_t>(input);
    if(!theta||oldk>kmax||kmax>g.n)throw runtime_error("invalid SSR dimensions");
    auto chosen=read_vec<int32_t>(input,oldk);PythonMT rng(input);Sampler sampler(g);
    vector<uint32_t> roots(theta);vector<uint8_t> covered(theta),available(g.n,1);
    vector<vector<uint32_t>> oldgains(oldk,vector<uint32_t>(g.n));
    for(uint32_t t=0;t<theta;++t) {
        auto root=roots[t]=rng.below(g.n);bool hit=false;
        for(uint32_t j=0;j<oldk;++j) {
            const auto& rr=sampler.sample(root,rng);
            if(!hit) {
                for(auto v:rr) ++oldgains[j][v];
                hit=find(rr.begin(),rr.end(),chosen[j])!=rr.end();
            }
        }
        covered[t]=hit;
    }
    vector<uint32_t> gains_chosen;
    for(uint32_t j=0;j<oldk;++j) {
        int32_t best=-1;for(uint32_t v=0;v<g.n;++v)if(available[v]&&(best<0||oldgains[j][v]>oldgains[j][best]))best=v;
        if(best!=chosen[j])throw runtime_error("original SSR prefix mismatch at round "+to_string(j));
        available[best]=0;gains_chosen.push_back(oldgains[j][best]);
    }
    oldgains.clear();oldgains.shrink_to_fit();
    double replay_seconds=elapsed(start);auto ext_start=Clock::now();
    cerr<<"original "<<oldk<<" SSR selections exactly reproduced in "<<replay_seconds<<"s"<<endl;
    vector<vector<uint32_t>> index(g.n);
    for(uint32_t j=oldk;j<kmax;++j) {
        PythonMT extra(input);for(auto& row:index)row.clear();
        for(uint32_t t=0;t<theta;++t) {
            const auto& rr=sampler.sample(roots[t],extra);
            if(!covered[t])for(auto v:rr)if(available[v])index[v].push_back(t);
        }
        int32_t best=-1;for(uint32_t v=0;v<g.n;++v)if(available[v]&&(best<0||index[v].size()>index[best].size()))best=v;
        chosen.push_back(best);available[best]=0;gains_chosen.push_back(index[best].size());
        for(auto t:index[best])covered[t]=1;
        if((j+1)%50==0)cerr<<"SSR "<<j+1<<"/"<<kmax<<" components, extension "<<elapsed(ext_start)<<"s"<<endl;
    }
    out<<setprecision(17)<<"{\"selected\":";json_array(out,chosen);out<<",\"selected_marginal_counts\":";json_array(out,gains_chosen);
    out<<",\"prefix_reproduced\":true,\"replay_seconds\":"<<replay_seconds<<",\"extension_seconds\":"<<elapsed(ext_start)<<",\"stats\":";sampler.stats(out);
    out<<",\"covered_count\":"<<count(covered.begin(),covered.end(),uint8_t(1))<<'}';
}


void run_sensitivity_bank(Graph& g,istream& input,ostream& out) {
    const auto begin=Clock::now();
    uint32_t theta=read_one<uint32_t>(input),k=read_one<uint32_t>(input);
    if(!theta||!k||k>g.n)throw runtime_error("invalid dimensions");
    PythonMT rng(input);Sampler sampler(g);
    vector<vector<vector<uint32_t>>> index(k,vector<vector<uint32_t>>(g.n));
    uint64_t empty=0;
    for(uint32_t t=0;t<theta;++t) {
        uint32_t root=rng.below(g.n);bool any=false;
        for(uint32_t j=0;j<k;++j) {
            const auto& rr=sampler.sample(root,rng);any=any||!rr.empty();
            for(auto v:rr)index[j][v].push_back(t);
        }
        if(!any)++empty;
    }
    double sampling_seconds=elapsed(begin);const auto greedy_begin=Clock::now();
    vector<uint8_t> available(g.n,1),covered(theta);
    vector<int32_t> chosen;vector<uint64_t> chosen_gains;
    for(uint32_t j=0;j<k;++j) {
        int32_t best=-1;uint64_t best_gain=0;
        for(uint32_t v=0;v<g.n;++v)if(available[v]) {
            uint64_t gain=0;for(auto t:index[j][v])if(!covered[t])++gain;
            if(best<0||gain>best_gain) {best=v;best_gain=gain;}
        }
        chosen.push_back(best);chosen_gains.push_back(best_gain);available[best]=0;
        for(auto t:index[j][best])covered[t]=1;
    }
    const auto total_covered=count(covered.begin(),covered.end(),uint8_t(1));
    out<<setprecision(17)<<"{\"selected\":";json_array(out,chosen);
    out<<",\"selected_marginal_counts\":";json_array(out,chosen_gains);
    out<<",\"sampling_seconds\":"<<sampling_seconds<<",\"greedy_seconds\":"<<elapsed(greedy_begin)
       <<",\"training_estimate_not_quality\":"<<double(g.n)*total_covered/theta
       <<",\"covered_count\":"<<total_covered<<",\"empty_ssr_count\":"<<empty
       <<",\"root_count_sum\":"<<theta<<",\"stats\":";sampler.stats(out);out<<'}';
}


int main(int argc,char** argv) {
    try {
        if(argc!=5)throw runtime_error("usage: kernel mode graph.bin input.bin output");
        ifstream graphfile(argv[2],ios::binary),input(argv[3],ios::binary);ofstream output(argv[4],ios::binary);Graph graph(graphfile);
        string mode=argv[1];
        if(mode=="ssrbank")run_sensitivity_bank(graph,input,output);
        else if(mode=="extend")run_ssr(graph,input,output);
        else if(mode=="forward")run_forward(graph,input,output);
        else if(mode=="evaluate")run_evaluation(graph,input,output);
        else throw runtime_error("unknown mode");
        if(!output)throw runtime_error("write failed");
    } catch(const exception& e) {cerr<<e.what()<<endl;return 1;}
    return 0;
}
