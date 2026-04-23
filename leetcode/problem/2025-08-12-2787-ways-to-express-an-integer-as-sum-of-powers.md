---
layout: leetcode-entry
title: "2787. Ways to Express an Integer as Sum of Powers"
permalink: "/leetcode/problem/2025-08-12-2787-ways-to-express-an-integer-as-sum-of-powers/"
leetcode_ui: true
entry_slug: "2025-08-12-2787-ways-to-express-an-integer-as-sum-of-powers"
---

[2787. Ways to Express an Integer as Sum of Powers](https://leetcode.com/problems/ways-to-express-an-integer-as-sum-of-powers/description/) medium
[blog post](https://leetcode.com/problems/ways-to-express-an-integer-as-sum-of-powers/solutions/7070952/kotlin-rust-by-samoylenkodmitry-xovl/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/12082025-2787-ways-to-express-an?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/noJl2ocdTsE)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1078

#### Problem TLDR

Ways to make n by sum of a^x+b^x+... #medium #dp

#### Intuition

Solved not optimally.
Consider all numbers: `p = 1, 2, 3, ... a, b, c ... n` and their x-powers: `pow = a^x b^x c^x`.
Do depth-first search, at each step make a decision:
* take the number: `v + a^x, p++`
* or skip it: `v, p++`
If we arrive at `v==n` we have a one good combination, return `1`.
Cache the answer for inputs `v, p`.

For optimization, rewrite to iterate over the same values `v, p`, reversing the ranges.
Then do a space optimization, as we always look at `p+1` previous row.

#### Approach

* I guess it is always good to start with simple choice instead of range iteration inside DFS

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 1092ms
    fun numberOfWays(n: Int, x: Int): Int {
        val m = 1000000007; var maxp = n; val pow = IntArray(n + 1) { 1 }
        for (i in pow.indices) for (j in 1..x) {
            pow[i] *= i; if (pow[i] > n) maxp = min(maxp, i)
        }
        val ways = Array(maxp + 2) { IntArray(n + 1) }
        for (i in ways.indices) ways[i][0] = 1
        for (x in 1..n) for (from in maxp downTo 1)
            for (y in from..maxp) {
                if (x - pow[y] < 0) break
                ways[from][x] = (ways[from][x] + ways[y + 1][x - pow[y]])%m
            }
        return ways[1][n]
    }

```
```kotlin

// 657ms
    fun numberOfWays(n: Int, x: Int): Int {
        val m = 1000000007; val dp = HashMap<Pair<Int, Int>, Int>()
        fun dfs(v: Int, p: Int): Int = dp.getOrPut(v to p) {
            if (v == 0) 1 else if (v < 0 || p > v || Math.pow(1.0*p, 1.0*x).toInt() > v) 0
            else (dfs(v - Math.pow(1.0*p, 1.0*x).toInt(), p + 1) + dfs(v, p + 1)) % m
        }
        return dfs(n, 1)
    }

```
```kotlin

// 15ms
    fun numberOfWays(n: Int, x: Int): Int {
        val m = 1000000007; val ways = IntArray(n + 1); ways[0] = 1
        for (p in 1..n) {
            var pow = 1; for (i in 1..x) pow *= p; if (pow > n) break
            for (v in n downTo pow) ways[v] = (ways[v] + ways[v - pow]) % m
        }
        return ways[n]
    }

```
```rust

// 0ms
    pub fn number_of_ways(n: i32, x: i32) -> i32 {
        let mut dp = [0;301]; let n = n as usize; dp[0] = 1;
        for p in 1..=n {
            let pow = p.pow(x as u32);
            for v in (pow..=n).rev() { dp[v] = (dp[v] + dp[v - pow]) % 1000000007 }
        } dp[n]
    }

```
```c++

// 14ms
    int numberOfWays(int n, int x) {
        int d[301]={}; d[0] = 1;
        for (int p = 1; p <= n && pow(p, x) <= n; ++p)
            for (int v = n, pw = pow(p, x); v >= pw; --v)
                d[v] = (d[v] + d[v-pw]) % 1000000007;
        return d[n];
    }

```
```python

// 363ms
    def numberOfWays(self, n: int, x: int) -> int:
        d = [1] + [0] * n
        for p in range(1, n + 1):
            for v in range(n, p ** x - 1, -1):
                d[v] = (d[v] + d[v - p ** x]) % 1000000007
        return d[n]

```

