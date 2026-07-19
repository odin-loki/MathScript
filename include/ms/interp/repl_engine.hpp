#pragma once

#include "ms/core/matrix.hpp"
#include "ms/error/expected.hpp"
#include "ms/frameworks/cellai/cellai.hpp"
#include "ms/frameworks/cypha/cypha.hpp"
#include "ms/frameworks/izaac/izaac.hpp"
#include "ms/tensorops/tensorops.hpp"
#include <atomic>
#include <map>
#include <optional>
#include <ostream>
#include <string>
#include <variant>
#include <vector>

namespace ms::interp {

struct PlotSeries {
    enum class Kind { Line, Bar, Scatter, Heatmap, Spy, Surface3D };

    std::vector<double> x;
    std::vector<double> y;
    Kind kind = Kind::Line;
    bool valid = false;
    size_t matrix_rows = 0;
    size_t matrix_cols = 0;
    size_t nnz = 0;
    Matrix<double> grid;
};

struct SessionState {
    std::map<std::string, Matrix<double>> matrices;
    std::map<std::string, double> scalars;
    std::vector<std::string> history;
    PlotSeries plot;
};

struct ScalarOperand {
    bool is_literal = true;
    double literal = 0.0;
    std::string name;
};

struct ScalarBinaryAssign {
    std::string target;
    char op = '+';
    ScalarOperand left;
    ScalarOperand right;
};

struct MatrixCallAssign {
    std::string target;
    std::string callee;
    std::vector<std::string> args;
};

struct ScalarMatrixCallAssign {
    std::string target;
    std::string callee;
    std::string arg;
    std::string arg2;
};

struct MultiMatrixCallAssign {
    std::vector<std::string> targets;
    std::string callee;
    std::string arg;
    std::vector<std::string> args;
};

using SessionObject = std::variant<
    izaac::bloom::BloomFilter,
    izaac::ratelimit::TokenBucket,
    cellai::CellMemory,
    cypha::DifModel,
    izaac::consensus::Cluster,
    tensorops::CPDecomposition,
    tensorops::TuckerDecomposition>;

class Interpreter {
public:
    Result<std::string> execute(const std::string& line);
    void set_cancel_flag(std::atomic<bool>* flag);
    bool cancel_requested() const;
    const SessionState& state() const { return state_; }
    bool has_plot() const { return state_.plot.valid; }
    const PlotSeries& plot() const { return state_.plot; }
    void reset() {
        state_ = SessionState{};
        session_objects_.clear();
    }
    Result<void> save_session(const std::string& path) const;
    Result<void> load_session(const std::string& path);
    Result<std::string> run_file(const std::string& path);
    Result<void> export_history(const std::string& path) const;
    static std::string trim(std::string s);
    static bool is_script_skip_line(const std::string& line);
    static bool try_parse_scalar_assignment(const std::string& line, std::string& name, double& value);
    static bool try_parse_scalar_binary_assignment(const std::string& line, ScalarBinaryAssign& assign);
    static bool try_parse_scalar_expr_assignment(const std::string& line, std::string& name, std::string& expr);
    static bool try_parse_matrix_call_assignment(const std::string& line, MatrixCallAssign& assign);
    static bool try_parse_scalar_matrix_call_assignment(const std::string& line, ScalarMatrixCallAssign& assign);
    static bool try_parse_multi_matrix_call_assignment(const std::string& line, MultiMatrixCallAssign& assign);
    static std::vector<std::string> list_scalar_expr_variables(const std::string& expr);
    static Result<double> eval_scalar_op(char op, double left, double right);
    static Result<double> eval_scalar_call(const std::string& name, const std::vector<double>& args);
    Result<std::string> assign_scalar(const std::string& name, double value);
    Result<std::string> assign_scalar_binary(const ScalarBinaryAssign& assign);
    Result<std::string> assign_scalar_expr(const std::string& name, const std::string& expr);
    Result<std::string> assign_matrix_call(const MatrixCallAssign& assign);
    Result<Matrix<double>> assign_matrix_call_tail(const MatrixCallAssign& assign);
    Result<Matrix<double>> assign_matrix_call_tail2(const MatrixCallAssign& assign);
    Result<Matrix<double>> assign_matrix_call_tail3(const MatrixCallAssign& assign);
    Result<Matrix<double>> assign_matrix_call_tail4(const MatrixCallAssign& assign);
    Result<Matrix<double>> assign_matrix_call_tail5(const MatrixCallAssign& assign);
    Result<Matrix<double>> assign_matrix_call_tail6(const MatrixCallAssign& assign);
    Result<Matrix<double>> assign_matrix_call_tail7(const MatrixCallAssign& assign);
    Result<std::string> assign_scalar_matrix_call(const ScalarMatrixCallAssign& assign);
    Result<std::string> assign_multi_matrix_call(const MultiMatrixCallAssign& assign);
    std::vector<std::pair<std::string, std::string>> list_session_objects() const;

private:
    SessionState state_;
    std::map<std::string, SessionObject> session_objects_;
    std::atomic<bool>* cancel_flag_ = nullptr;
    std::optional<Result<std::string>> try_session_object_command(const std::string& cmd);
    Result<Matrix<double>> parse_matrix(const std::string& text) const;
    Result<Matrix<double>> resolve_matrix(const std::string& name) const;
    Result<Matrix<double>> eval_matrix_operand(const std::string& text);
    Result<std::string> set_plot(const Matrix<double>& xs, const Matrix<double>& ys,
                                 PlotSeries::Kind kind = PlotSeries::Kind::Line);
    Result<std::string> set_plot_bars(std::vector<double> x, std::vector<double> y);
    Result<std::string> set_plot_heatmap(const Matrix<double>& m);
    Result<std::string> set_plot_spy(const Matrix<double>& m);
    Result<std::string> set_plot_surf(const Matrix<double>& z);
    Result<std::string> set_plot_surf(const Matrix<double>& x, const Matrix<double>& y,
                                      const Matrix<double>& z);
};

void print_matrix(std::ostream& out, const Matrix<double>& m);

bool repl_cancel_requested();

} // namespace ms::interp
