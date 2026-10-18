#pragma once
// Compiled execution of the existing coupon sampler. RNG state is supplied
// by Python random.Random; random() and randrange() reproduce its draw stream.
// No fast-math: probabilities and interval endpoints come from CouponRR.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
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

