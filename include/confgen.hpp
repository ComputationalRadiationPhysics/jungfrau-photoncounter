#pragma once

#include <ostream>
#include <type_traits>
#include <utility>

struct Empty {};

template <class Head_, class Tail_>
struct List {
    using Head = Head_;
    using Tail = Tail_;
};

template <class... Ts>
struct Tuple {};

template <class T, T value_>
struct Value {
    static constexpr T value = value_;
};

template <class H, class... Tail>
struct MakeList {
    using Result = List<H, typename MakeList<Tail...>::Result>;
};

template <class H>
struct MakeList<H> {
    using Result = List<H, Empty>;
};

template <class T, T... vs>
using Values = typename MakeList<Value<T, vs>...>::Result;

template <class X, class Ys>
struct RightProduct {
    using H = typename Ys::Head;
    using T = typename Ys::Tail;
    using Result = List<typename MakeList<X, H>::Result, typename RightProduct<X, T>::Result>;
};

template <class X>
struct RightProduct<X, Empty> {
    using Result = Empty;
};

template <class Xs, class Ys>
struct Append {
    using Result = List<typename Xs::Head, typename Append<typename Xs::Tail, Ys>::Result>;
};

template <class Ys>
struct Append<Empty, Ys> {
    using Result = Ys;
};

template <class Xs>
struct Associate {
    using Result = Xs;
};

template <class X, class Xs, class Ys>
struct Associate<List<List<X, Xs>, Ys>> {
    using H = List<X, Xs>;
    using T = Ys;
    using Result = typename Append<typename Associate<H>::Result, T>::Result;
};

template <class Xs>
struct AssociateAll {
    using Result = Xs;
};

template <>
struct AssociateAll<Empty> {
    using Result = Empty;
};

template <class X, class Xs>
struct AssociateAll<List<X, Xs>> {
    using H = X;
    using T = Xs;
    using Result = List<typename Associate<H>::Result, typename AssociateAll<T>::Result>;
};

template <class Xs, class Ys>
struct Cartesian {
    using H = typename Xs::Head;
    using T = typename Xs::Tail;
    using Result =
        typename AssociateAll<typename Append<typename RightProduct<H, Ys>::Result,
                                              typename Cartesian<T, Ys>::Result>::Result>::Result;
};

template <class Ys>
struct Cartesian<Empty, Ys> {
    using Result = Empty;
};

template <class Tuple, class T>
struct TupleCat;

template <class... Ts, class T>
struct TupleCat<Tuple<Ts...>, T> {
    using Result = Tuple<Ts..., T>;
};

template <class, class>
struct Flatten;

template <class Accu>
struct Flatten<Accu, Empty> {
    using Result = Accu;
};

template <class Accu, class H, class T>
struct Flatten<Accu, List<H, T>> {
    using Result = typename Flatten<typename TupleCat<Accu, H>::Result, T>::Result;
};

template <class X>
struct Print {
    using Show = typename X::printer__;
};

template <class T, T v>
std::ostream& operator<<(std::ostream& os, Value<T, v>) {
    return os << v;
}

template <class H, class T>
void printList(std::ostream& os, List<H, T> xs) {
    os << H{} << ", ";
    printList(os, T{});
}

template <class H>
void printList(std::ostream& os, List<H, Empty>) {
    os << H{};
}

template <class H, class T>
std::ostream& operator<<(std::ostream& os, List<H, T> xs) {
    os << "[";
    printList(os, xs);
    os << "]";
    return os;
};

using Test = Values<int, 1, 2>;

using Xs = Cartesian<Test, Test>::Result;
using Ys = Cartesian<Xs, Test>::Result;
using Zs = Cartesian<Ys, Test>::Result;
using Ws = Cartesian<Zs, Test>::Result;

using Fs = Flatten<Tuple<>, Ws>::Result;

using Ts = Ws::Head;

template <class... Ts>
constexpr std::size_t length(Tuple<Ts...>) {
    return sizeof...(Ts);
}

template <class... Ts, class... Us>
constexpr auto operator*(const List<Ts...>&, const List<Us...>&) ->
    typename Cartesian<List<Ts...>, List<Us...>>::Result {
    return {};
}

template <int, class>
struct Get;

template <class Head, class... Tail>
struct Get<0, Tuple<Head, Tail...>> {
    using Result = Head;
};

template <int i, class Head, class... Tail>
struct Get<i, Tuple<Head, Tail...>> {
    using Result = typename Get<i - 1, Tuple<Tail...>>::Result;
};

template <int i, class Tuple>
using Get_t = typename Get<i, Tuple>::Result;
