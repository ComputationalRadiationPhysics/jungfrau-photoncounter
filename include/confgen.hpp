#pragma once

#include <iostream>
#include <type_traits>

template <class...>
struct List;

template <>
struct List<> {};

template <class T, class... Ts>
struct List<T, Ts...> : List<Ts...> {
    T value;
    constexpr List(const List &) = default;
    constexpr List(const T &head, const List<Ts...> &tail) : List<Ts...>{tail}, value{head} {}
    template <class U, class... Us>
    constexpr List(const U &head, const Us &... tail) : List<Ts...>{tail...}, value{head} {}
};

template <class T, class... Ts>
constexpr auto cons(T head, const List<Ts...> &tail) -> List<T, Ts...> {
    return {head, tail};
}

template <class F>
constexpr auto fmap(F &&, const List<> &) -> List<> {
    return {};
}

template <class F, class T, class... Ts>
constexpr auto fmap(F &&f, const List<T, Ts...> &xs) {
    return cons(f(xs.value), fmap(std::forward<F>(f), static_cast<const List<Ts...> &>(xs)));
}

template <class... Us>
constexpr auto operator+(const List<> &a, const List<Us...> &b) -> List<Us...> {
    return b;
}

template <class... Ts>
constexpr auto operator+(const List<Ts...> &a, const List<> &b) -> List<Ts...> {
    return a;
}

template <class T, class... Ts, class U, class... Us>
constexpr auto operator+(const List<T, Ts...> &a, const List<U, Us...> &b)
    -> List<T, Ts..., U, Us...> {
    return cons(a.value, (static_cast<const List<Ts...> &>(a) + b));
}

template <int I, class T, class... Ts>
struct ListIterator {
    using type = typename ListIterator<I - 1, Ts...>::type;
};

template <class T, class... Ts>
struct ListIterator<0, T, Ts...> {
    using type = List<T, Ts...>;
};

template <int I, class... Ts>
constexpr auto &get(const List<Ts...> &xs) {
    using T = typename ListIterator<I, Ts...>::type;
    return static_cast<const T &>(xs).value;
}

template <class... Ts>
constexpr auto makeList(Ts &&... args) -> List<std::remove_cv_t<std::remove_reference_t<Ts>>...> {
    return {args...};
}

template <class... Ts>
constexpr auto zip(const List<Ts...> &a, const List<> &b) -> List<> {
    return {};
}

template <class... Us>
constexpr auto zip(const List<> &a, const List<Us...> &b) -> List<> {
    return {};
}

template <class T, class... Ts, class U, class... Us>
constexpr auto zip(const List<T, Ts...> &a, const List<U, Us...> &b) {
    using TailA = List<Ts...>;
    using TailB = List<Us...>;
    return cons(List<T, U>{a.value, b.value},
                zip(static_cast<const TailA &>(a), static_cast<const TailB &>(b)));
}

template <class T>
constexpr auto mulRight(const T &, const List<> &) -> List<> {
    return {};
}

template <class T, class U, class... Us>
constexpr auto mulRight(const T &x, const List<U, Us...> &ys) {
    using Tail = List<Us...>;
    return cons(List<T, U>{x, ys.value}, mulRight(x, static_cast<const Tail &>(ys)));
}

template <class... Us>
constexpr auto cartesian(const List<> &, const List<Us...> &) -> List<> {
    return {};
}

template <class T, class... Ts, class U, class... Us>
constexpr auto cartesian(const List<T, Ts...> &xs, const List<U, Us...> &ys) {
    using TailX = List<Ts...>;
    using TailY = List<Us...>;
    return cons(mulRight(xs.value, ys), cartesian(static_cast<const TailX &>(xs), ys));
}

struct Associator {
    template <class T, class... Ts>
    constexpr auto operator()(const List<List<Ts...>, T> &xs) -> List<Ts..., T> const {
        return concat(cons(xs.value, makeList(static_cast<const List<T> &>(xs))));
    }
    template <class... Ts>
    constexpr auto operator()(const List<Ts...> &xs) -> List<Ts...> const {
        return xs;
    }
};

template <class... Ts, class... Us>
constexpr auto operator*(const List<Ts...> &xs, const List<Us...> &ys) {
    return fmap(Associator{}, concat(cartesian(xs, ys)));
}

inline constexpr auto concat(const List<> &) -> List<> { return {}; }

template <class T, class... Ts, class... Us>
constexpr auto concat(const List<List<T, Ts...>, Us...> &xs) {
    return xs.value + concat(static_cast<const List<Us...> &>(xs));
}

template <class T, int N>
struct Array {
    T __data[N];
    constexpr T &operator[](int k) { return __data[N]; }
    constexpr const T &operator[](int k) const { return __data[N]; }
    int size() const { return N; }
};

template <class T>
struct Type {};

template <class T>
void printList(std::ostream &os, const List<T> &xs) {
    os << xs.value;
}

template <class T1, class T2, class... Ts>
void printList(std::ostream &os, const List<T1, T2, Ts...> &xs) {
    using Tail = List<T2, Ts...>;
    os << xs.value << ", ";
    printList(os, static_cast<const Tail &>(xs));
}

template <class T, class... Ts>
std::ostream &operator<<(std::ostream &os, const List<T, Ts...> &xs) {
    os << "[";
    printList(os, xs);
    return os << "]";
}

inline std::ostream &operator<<(std::ostream &os, const List<> &) { return os << "[]"; }
