// Compiled execution of the existing coupon sampler. RNG state is supplied
// by Python random.Random; random() and randrange() reproduce its draw stream.
// No fast-math: probabilities and interval endpoints come from CouponRR.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace std;
using Clock = chrono::steady_clock;
double elapsed(Clock::time_point start) { return chrono::duration<double>(Clock::now()-start).count(); }
template<class T> T read_one(istream& f) { T v; f.read(reinterpret_cast<char*>(&v),sizeof(v)); if(!f) throw runtime_error("short input"); return v; }
template<class T> vector<T> read_vec(istream& f,size_t n) { vector<T> v(n); f.read(reinterpret_cast<char*>(v.data()),n*sizeof(T)); if(!f) throw runtime_error("short array"); return v; }
template<class T> void write_one(ostream& f,T v) { f.write(reinterpret_cast<char*>(&v),sizeof(v)); }
template<class T> void json_array(ostream& out,const vector<T>& a) { out<<'['; for(size_t i=0;i<a.size();++i) { if(i)out<<','; out<<a[i]; } out<<']'; }

struct PythonMT {
    array<uint32_t,624> mt; uint32_t index;
    explicit PythonMT(istream& f) { for(auto& x:mt) x=read_one<uint32_t>(f); index=read_one<uint32_t>(f); if(index>624)throw runtime_error("bad RNG state"); }
    uint32_t next() {
        if(index>=624) {
            for(int i=0;i<624;++i) {
                uint32_t y=(mt[i]&0x80000000U)|(mt[(i+1)%624]&0x7fffffffU);
                mt[i]=mt[(i+397)%624]^(y>>1)^((y&1)?0x9908b0dfU:0U);
            }
            index=0;
        }
        uint32_t y=mt[index++]; y^=y>>11; y^=(y<<7)&0x9d2c5680U; y^=(y<<15)&0xefc60000U; y^=y>>18; return y;
    }
    double uniform() { uint32_t a=next()>>5,b=next()>>6; return (a*67108864.0+b)/9007199254740992.0; }
    uint32_t below(uint32_t n) {
        if(!n)throw runtime_error("empty range");
        unsigned bits=0; for(uint32_t x=n;x;x>>=1)++bits;
        uint32_t r; do {r=next()>>(32-bits);}while(r>=n); return r;
    }
};

struct Graph {
    uint32_t n,m; vector<uint64_t> ptr,inptr,inedges; vector<int32_t> edges,sources;
    vector<double> a,d,low,high;
    explicit Graph(istream& f) {
        n=read_one<uint32_t>(f);m=read_one<uint32_t>(f);
        ptr=read_vec<uint64_t>(f,n+1); edges=read_vec<int32_t>(f,m);
        a=read_vec<double>(f,n); d=read_vec<double>(f,n);
        inptr=read_vec<uint64_t>(f,n+1); inedges=read_vec<uint64_t>(f,m);
        sources=read_vec<int32_t>(f,m);low=read_vec<double>(f,m);high=read_vec<double>(f,m);
    }
};

struct Sampler {
    const Graph& g; vector<uint64_t> cached,member; vector<double> draws; vector<int32_t> queue;
    uint64_t epoch=0,calls=0,nonempty=0,memberships=0,scanned=0,draw_count=0,max_size=0;
    explicit Sampler(const Graph& graph):g(graph),cached(g.n),member(g.n),draws(g.n) {}
    const vector<int32_t>& sample(uint32_t root,PythonMT& rng) {
        ++epoch; ++calls; queue.clear(); draws[root]=rng.uniform();cached[root]=epoch;++draw_count;
        if(draws[root]>=g.a[root])return queue;
        queue.push_back(root);member[root]=epoch;++nonempty;
        for(size_t q=0;q<queue.size();++q) {
            auto v=queue[q];scanned+=g.inptr[v+1]-g.inptr[v];
            for(auto pos=g.inptr[v];pos<g.inptr[v+1];++pos) {
                auto e=g.inedges[pos];auto w=g.sources[e];
                if(cached[w]!=epoch) {cached[w]=epoch;draws[w]=rng.uniform();++draw_count;}
                if(g.low[e]<=draws[w] && draws[w]<g.high[e] && member[w]!=epoch) {member[w]=epoch;queue.push_back(w);}
            }
        }
        memberships+=queue.size();max_size=max(max_size,uint64_t(queue.size()));return queue;
    }
    void stats(ostream& out) {
        out<<"{\"calls\":"<<calls<<",\"nonempty\":"<<nonempty<<",\"memberships\":"<<memberships
           <<",\"scanned_edges\":"<<scanned<<",\"draw_count\":"<<draw_count<<",\"max_size\":"<<max_size<<'}';
    }
};

struct Walker {
    const Graph& g;vector<uint64_t> visited;uint64_t epoch=0;
    explicit Walker(const Graph& graph):g(graph),visited(g.n) {}
    int32_t walk(int32_t start,PythonMT& rng) {
        ++epoch;auto v=start;
        while(visited[v]!=epoch) {
            visited[v]=epoch;double r=rng.uniform();
            if(r<g.a[v])return v;
            if(r<g.a[v]+g.d[v])return -1;
            uint32_t degree=uint32_t(g.ptr[v+1]-g.ptr[v]);if(!degree)return -1;
            v=g.edges[g.ptr[v]+rng.below(degree)];
        }
        return -1;
    }
};

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

void run_forward(Graph& g,istream& input,ostream& out) {
    auto batches=read_one<uint32_t>(input);PythonMT rng(input);Walker walker(g);
    for(uint32_t v=0;v<g.n;++v) {
        vector<int64_t> outcomes;
        for(uint32_t b=0;b<batches;++b) {auto u=walker.walk(v,rng);if(u>=0)outcomes.push_back(int64_t(b)*g.n+u);}
        write_one(out,uint32_t(outcomes.size()));out.write(reinterpret_cast<char*>(outcomes.data()),outcomes.size()*sizeof(int64_t));
    }
}

void run_evaluation(Graph& g,istream& input,ostream& out) {
    uint32_t size=read_one<uint32_t>(input),trials=read_one<uint32_t>(input);Walker walker(g);vector<uint64_t> consumed(g.n);uint64_t epoch=0;
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
        if(mode=="ssr")run_ssr(graph,input,output);
        else if(mode=="forward")run_forward(graph,input,output);
        else if(mode=="evaluate")run_evaluation(graph,input,output);
        else throw runtime_error("unknown mode");
        if(!output)throw runtime_error("write failed");
    } catch(const exception& e) {cerr<<e.what()<<endl;return 1;}
    return 0;
}
