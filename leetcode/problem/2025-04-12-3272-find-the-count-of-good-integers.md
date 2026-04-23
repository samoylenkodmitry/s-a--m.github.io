---
layout: leetcode-entry
title: "3272. Find the Count of Good Integers"
permalink: "/leetcode/problem/2025-04-12-3272-find-the-count-of-good-integers/"
leetcode_ui: true
entry_slug: "2025-04-12-3272-find-the-count-of-good-integers"
---

[3272. Find the Count of Good Integers](https://leetcode.com/problems/find-the-count-of-good-integers/description/) hard
[blog post](https://leetcode.com/problems/find-the-count-of-good-integers/solutions/6642761/kotlin-rust-by-samoylenkodmitry-ah9f/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/12042025-3272-find-the-count-of-good?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/VAQFiJoxLeA)
![1.webp](/assets/leetcode_daily_images/c7a539a5.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/956

#### Problem TLDR

Count palindromes of length n, divisible by k #hard #math #permutations

#### Intuition

Failed this.

My intuition was
1. build halves of palindromes
2. take those % k == 0
3. count permutations of them

The problems:
1. some palindromes were in the same permutations buckets
2. my permutations counting function was wrong

I was able to write TLE solution.

```j

    // n=2  k=1
    // 10 11 12...19
    // 20 21 22...29
    // .............
    // 90 91 92...99

    // 0 1 2 3 4 5 6 7 9
    // *   *   *   *     k=2
    //       *           k=3   19 % 3 != 0
    // try to build all of them? and prune early
    // shold be palindrome -> or rearranged to it
    // palindrome should be % k == 0
    // 10^n
    // how to rearrange? (to palindrome)

    // palindrome properties:
    // all digits in pairs (except middle)
    // we have exactly n digits
    //   a b c m c b a   -> aa bb cc m
    // ok, we have sets of n/2 digits
    // check if any set permutaion is % k == 0
    // if so, we add entire set permutation count of n
    // permutation count p(n) = n * p(n - 1) (? check this)

    // how to handle avoiding `0` at start ?
    // append digits in an increasing order
    // 0
    //
    // 1  11  111
    // 12 122 1222
    //    112 1122 11222
    //        1112 11122 111222

```

Now, looking at the `u/lee215/`'s solution:
1. I should have known the permutations with duplicates formula: `pd(n) = p(n) / p(d)`. Meaning, to remove duplicate, we dividing by permutations of count of them.
2. I should have known permutations `except first position` formula: `p0(n) = (n - count_zeros) * p(n - 1)`. Meaning, we holding the first position, and left with `n-1` other positions. We have exactly `count_zeros` events where each zero can be at the first position.

#### Approach

* combinatorics is hard

#### Complexity

- Time complexity:
$$O(10^n)$$, the range is `9 * 10^(n/2)`

- Space complexity:
$$O(n!)$$, unique permutations of size `n` are stored in a set

#### Code

```kotlin

    fun countGoodIntegers(n: Int, k: Int): Long {
        val seen = HashSet<Map<Char, Int>>(); var res = 0L; val h = 1.0 * ((n - 1) / 2)
        val l = Math.pow(10.0, h).toInt(); fun f(n: Int): Int = if (n < 2) 1 else n * f(n - 1)
        return (l..<l * 10).sumOf {
            val s = "$it" + "$it".dropLast(n % 2).reversed()
            if (s.toLong() % k.toLong() > 0L) return@sumOf 0L
            val cnt = s.groupingBy { it }.eachCount(); if (!seen.add(cnt)) return@sumOf 0L
            1L * (n - (cnt['0'] ?: 0)) * f(n - 1) / cnt.values.fold(1L) { r, c -> r * f(c)}
        }
    }

```
```rust

    pub fn count_good_integers(n: i32, k: i32) -> i64 {
        let (mut s, mut ans, n) = (HashSet::new(), 0, n as usize);
        let lo = 10_usize.pow((n as u32 - 1) / 2);
        let f = [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800];
        for p in lo..lo * 10 {
            let (mut freq, mut x) = ([0; 10], p);
            freq[p % 10] += n & 1; let mut t = p / (1 + 9 * (n & 1));
            while t > 0 { let d = t % 10; freq[d] += 2; x = x * 10 + d; t /= 10; }
            if x % (k as usize) != 0 || !s.insert(freq) { continue; }
            let mut num = (n - freq[0]) * f[n - 1]; for x in freq { if x > 1 { num /= f[x] }}
            ans += num
        }
        ans as _
    }

```
```c++

    long long countGoodIntegers(int n, int k) {
        set<array<int, 10>> seen; long long ans = 0;
        int hi = 1, f[] = { 1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800 };
        for (int i = 0; i < (n + 1) / 2; ++i) hi *= 10;
        for (int p = hi / 10; p < hi; ++p) {
            array<int, 10> freq{}; freq[p % 10] += n & 1; long long x = p;
            for (int t = p / (1 + 9 * (n & 1)); t; t /= 10)
                freq[t % 10] += 2, x = x * 10 + t % 10;
            if (x % k || !seen.insert(freq).second) continue;
            long long num = (n - freq[0]) * f[n - 1]; for (int x: freq) if (x > 1) num /= f[x];
            ans += num;
        }
        return ans;
    }

```

