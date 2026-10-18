// Same row-major SSR bank and Python random stream as uniform_ssr.sample_ssr.
// Reuse the already verified cached-node RR sampler and forward evaluator.
#define main figure3_unused_entrypoint
#include "figure3_extension_kernel.cpp"
#undef main

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
        if(argc!=5)throw runtime_error("usage: kernel ssrbank|evaluate graph.bin input.bin output");
        ifstream graphfile(argv[2],ios::binary),input(argv[3],ios::binary);
        ofstream output(argv[4],ios::binary);Graph graph(graphfile);
        string mode=argv[1];
        if(mode=="ssrbank")run_sensitivity_bank(graph,input,output);
        else if(mode=="evaluate")run_evaluation(graph,input,output);
        else throw runtime_error("unknown mode");
        if(!output)throw runtime_error("write failed");
    }catch(const exception& e){cerr<<e.what()<<endl;return 1;}
    return 0;
}
