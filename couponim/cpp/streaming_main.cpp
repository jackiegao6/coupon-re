#include "memory.hpp"
#include "coupon_core.hpp"

int main(int argc, char** argv) {
    try {
        if (argc != 5) throw runtime_error("usage: stream graph.bin input.bin output-base prefix-budgets");
        auto graph_started = Clock::now();
        ifstream graphfile(argv[1], ios::binary), input(argv[2], ios::binary);
        Graph g(graphfile);
        double graph_load_seconds = elapsed(graph_started);
        auto started = Clock::now();
        uint32_t theta = read_one<uint32_t>(input), kmax = read_one<uint32_t>(input);
        if (!theta || !kmax || kmax > g.n) throw runtime_error("invalid dimensions");
        vector<uint32_t> targets;
        string text(argv[4]); size_t pos = 0;
        while (pos < text.size()) {
            size_t end = text.find(',', pos);
            auto k = uint32_t(stoul(text.substr(pos, end - pos)));
            if (!k || k > kmax) throw runtime_error("invalid prefix");
            targets.push_back(k);
            if (end == string::npos) break;
            pos = end + 1;
        }
        PythonMT root_rng(input); Sampler sampler(g);
        vector<uint32_t> roots(theta);
        vector<uint8_t> covered(theta), available(g.n, 1);
        auto stage = Clock::now();
        for (auto& root : roots) root = root_rng.below(g.n);
        double root_seconds = elapsed(stage), sampling_seconds = 0, greedy_seconds = 0;
        vector<int32_t> chosen; vector<uint32_t> chosen_gains;
        uint64_t covered_count = 0, max_memberships = 0, max_capacity_bytes = 0;
        for (uint32_t j = 0; j < kmax; ++j) {
            PythonMT rng(input);
            stage = Clock::now();
            {
                // Destruction at the end of this scope releases every round's lists.
                vector<vector<uint32_t>> index(g.n);
                uint64_t memberships = 0;
                for (uint32_t t = 0; t < theta; ++t) {
                    const auto& rr = sampler.sample(roots[t], rng);
                    if (!covered[t]) for (auto v : rr) if (available[v]) {
                        index[v].push_back(t); ++memberships;
                    }
                }
                sampling_seconds += elapsed(stage);
                stage = Clock::now();
                uint64_t capacity_bytes = uint64_t(index.capacity()) * sizeof(vector<uint32_t>);
                int32_t best = -1;
                for (uint32_t v = 0; v < g.n; ++v) {
                    capacity_bytes += uint64_t(index[v].capacity()) * sizeof(uint32_t);
                    if (available[v] && (best < 0 || index[v].size() > index[best].size())) best = v;
                }
                max_memberships = max(max_memberships, memberships);
                max_capacity_bytes = max(max_capacity_bytes, capacity_bytes);
                chosen.push_back(best); available[best] = 0;
                chosen_gains.push_back(uint32_t(index[best].size()));
                for (auto t : index[best]) covered[t] = 1;
                covered_count += index[best].size();
            }
            greedy_seconds += elapsed(stage); // Includes releasing the current index.
            if (find(targets.begin(), targets.end(), j + 1) != targets.end()) {
                auto mem = memory_usage();
                string name = string(argv[3]) + ".k" + to_string(j + 1) + ".json";
                ofstream out(name + ".tmp");
                out << setprecision(17) << "{\"k\":" << j+1 << ",\"theta\":" << theta << ",\"selected\":";
                json_array(out, chosen);
                out << ",\"selected_marginal_counts\":"; json_array(out, chosen_gains);
                out << ",\"covered_count\":" << covered_count
                    << ",\"root_sampling_seconds\":" << root_seconds
                    << ",\"sampling_seconds\":" << sampling_seconds
                    << ",\"greedy_seconds\":" << greedy_seconds
                    << ",\"graph_load_seconds\":" << graph_load_seconds
                    << ",\"selection_elapsed_seconds\":" << elapsed(started)
                    << ",\"peak_private_bytes\":" << (mem.peak_private < 0 ? string("null") : to_string(mem.peak_private))
                    << ",\"peak_working_set_bytes\":" << mem.peak_working_set
                    << ",\"current_private_bytes\":" << (mem.current_private < 0 ? string("null") : to_string(mem.current_private))
                    << ",\"max_current_memberships\":" << max_memberships
                    << ",\"max_index_capacity_bytes\":" << max_capacity_bytes
                    << ",\"retained_component_count\":1,\"stats\":";
                sampler.stats(out); out << '}'; out.close();
                if (!out) throw runtime_error("checkpoint write failed");
                if (rename((name + ".tmp").c_str(), name.c_str()) != 0) throw runtime_error("checkpoint publish failed");
            }
            if ((j+1) % 25 == 0 || j+1 == kmax)
                cerr << "component=" << j+1 << '/' << kmax << " elapsed_seconds=" << elapsed(started)
                     << " peak_rss_mib=" << memory_usage().peak_working_set / 1048576.0 << endl;
        }
    } catch (const exception& e) { cerr << e.what() << endl; return 1; }
    return 0;
}
