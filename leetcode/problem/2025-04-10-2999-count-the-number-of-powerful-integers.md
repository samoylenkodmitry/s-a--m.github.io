---
layout: leetcode-entry
title: "2999. Count the Number of Powerful Integers"
permalink: "/leetcode/problem/2025-04-10-2999-count-the-number-of-powerful-integers/"
leetcode_ui: true
entry_slug: "2025-04-10-2999-count-the-number-of-powerful-integers"
---

[2999. Count the Number of Powerful Integers](https://leetcode.com/problems/count-the-number-of-powerful-integers/description/) hard
[blog post](https://leetcode.com/problems/count-the-number-of-powerful-integers/solutions/6635848/kotlin-rust-by-samoylenkodmitry-lbgo/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/10042025-2999-count-the-number-of?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/CUYhG1pO0Dg)
![1.webp](/assets/leetcode_daily_images/1516238d.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/954

#### Problem TLDR

Numbers in range start..finish, digits up to limit #hard #dp #math

#### Intuition

Completely failed.

```j

    // 200..12345 lim 3, s "33"
    // 33 133 | 233 333
    //   1133 2133 3133
    //        1233 2233 3233
    //            1333 2333 3333

    //
    // limit
    // limit^2
    // limit^3
    // ...
    // limit^(log10(finish) - s.length) - prefix - suffix
    //

```

I was able to write the generation code that prints out all numbers (TLE solution).
After looking at generated numbers, I saw some pattern:
* the numbers count grows in `limit * previous`, meaning after x steps we have `limit ^ x` numbers
* the numbers digits are less than `limit`, meaning we are in a `base limit + 1` system, like, for limit = 9, we are in decimal, for limit = 1 we are in a binary

After that I've looked at the hints and they tell me nothing.

After that I've looked for solutions.

So, for the `base limit` solution, we just have to convert `start` and `finish` to this base and count the diff: finish(base_limit) - start(base_limit).
The corner case is when digits are `out of base`. Then we have to make those digits as max as possible. And deal with tail suffix:
* if suffix is bigger than based tail, do `-1` (why?)

#### Approach

* still solution from somebody
* the DFS + memo solution is more doable if you didn't spot the `base` system

#### Complexity

- Time complexity:
$$O(log(finish))$$

- Space complexity:
$$O(log(finish))$$

#### Code

```kotlin

    fun numberOfPowerfulInt(start: Long, finish: Long, limit: Int, s: String): Long {
        fun cnt(x: String): Long {
            var lo = '0'; var hi = lo + limit;
            val max = x.map { if (it > hi) lo = hi; it.coerceIn(lo, hi) }.joinToString("")
            return max.dropLast(s.length).ifEmpty { "0" }.toLong(limit + 1) -
                if (max.takeLast(s.length).toLong() < s.toLong()) 1 else 0
        }
        return cnt("" + finish) - cnt("" + (start - 1))
    }

```
```kotlin

    fun numberOfPowerfulInt(start: Long, finish: Long, limit: Int, s: String): Long {
        fun cnt(x: String): Long {
            if (x.length < s.length) return 0
            if (x.length == s.length) return if (x >= s) 1 else 0
            var cnt = 0L; val plen = x.length - s.length
            for (i in 0..<plen) {
                if (x[i] - '0' > limit) return cnt + Math.pow(1.0 * limit + 1, 1.0 * plen - i).toLong()
                cnt += (x[i] - '0') * Math.pow(1.0 * limit + 1, 1.0 * plen - i - 1).toLong()
            }
            return if (x.takeLast(s.length) >= s) cnt + 1 else cnt
        }
        return cnt("" + finish) - cnt("" + (start - 1))
    }

```
```kotlin

    fun numberOfPowerfulInt(start: Long, finish: Long, limit: Int, s: String): Long {
        val dp = HashMap<Pair<Pair<Int, Boolean>, Boolean>, Long>()
        val hi = finish.toString(); val low = start.toString().padStart(hi.length, '0')
        fun dfs(i: Int, useLow: Boolean, useHi: Boolean): Long = if (i == low.length) 1
            else dp.getOrPut(i to useLow to useHi) {
                var l = if (useLow) low[i] - '0' else 0
                var h = if (useHi) hi[i] - '0' else 9
                if (i < hi.length - s.length) (l..min(h, limit)).sumOf { d ->
                    dfs(i + 1, useLow && d == l, useHi && d == h) }
                else {
                    var d = s[i - hi.length + s.length] - '0'
                    if (d !in l..min(h, limit)) 0L else
                        dfs(i + 1, useLow && d == l, useHi && d == h)
                }
            }
        return dfs(0, true, true)
    }

```
```rust

    pub fn number_of_powerful_int(start: i64, finish: i64, limit: i32, s: String) -> i64 {
        fn cnt(x: String, l: u32, s: &str) -> i64 {
            let mut lo = b'0'; let hi = lo + l as u8;
            let max: String = x.bytes().map(|c| {
                if c > hi { lo = hi; }; c.min(hi).max(lo) as char
            }).collect();
            let a = i64::from_str_radix(&max[..max.len().saturating_sub(s.len())], l + 1).unwrap_or(0);
            let b = i64::from_str_radix(&max[max.len().saturating_sub(s.len())..], l + 1).unwrap_or(0);
            a - (b < i64::from_str_radix(s, l + 1).unwrap()) as i64
        }
        cnt(finish.to_string(), limit as u32, &s) - cnt((start - 1).to_string(), limit as u32, &s)
    }

```
```c++

    long long numberOfPowerfulInt(long long start, long long finish, int limit, string s) {
        auto to_dec = [&](const std::string& t) {
            long long r = 0, b = limit + 1; for (char c : t) r = r * b + (c - '0');
            return r;
        };
        auto cnt = [&](std::string x) {
            char lo = '0', hi = lo + limit; int n = x.size(), m = s.size(), d = std::max(n - m, 0);
            for (char& c : x) { if (c > hi) lo = hi; c = std::clamp(c, lo, hi); }
            auto a = to_dec(d ? x.substr(0, d) : "0"); auto b = to_dec(x.substr(d));
            return a - (b < to_dec(s));
        };
        return cnt(std::to_string(finish)) - cnt(std::to_string(start - 1));
    }

```

