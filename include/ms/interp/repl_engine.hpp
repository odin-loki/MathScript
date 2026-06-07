#pragma once

#include "ms/core/matrix.hpp"
#include "ms/error/expected.hpp"
#include <map>
#include <ostream>
#include <string>
#include <vector>

namespace ms::interp {

struct PlotSeries {
    enum class Kind { Line, Bar };

    std::vector<double> x;
    std::vector<double> y;
    Kind kind = Kind::Line;
    bool valid = false;
};

struct SessionState {
    std::map<std::string, Matrix<double>> matrices;
    std::map<std::string, double> scalars;
    std::vector<std::string> history;
    PlotSeries plot;
};

class Interpreter {
public:
    Result<std::string> execute(const std::string& line);
    const SessionState& state() const { return state_; }
    bool has_plot() const { return state_.plot.valid; }
    const PlotSeries& plot() const { return state_.plot; }
    void reset() { state_ = SessionState{}; }
    Result<void> save_session(const std::string& path) const;
    Result<void> load_session(const std::string& path);
    static std::string trim(std::string s);

private:
    SessionState state_;
    Result<Matrix<double>> parse_matrix(const std::string& text) const;
    Result<Matrix<double>> resolve_matrix(const std::string& name) const;
    Result<std::string> set_plot(const Matrix<double>& xs, const Matrix<double>& ys);
    Result<std::string> set_plot_bars(std::vector<double> x, std::vector<double> y);
};

void print_matrix(std::ostream& out, const Matrix<double>& m);

} // namespace ms::interp
