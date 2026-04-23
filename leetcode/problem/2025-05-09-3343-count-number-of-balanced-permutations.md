---
layout: leetcode-entry
title: "3343. Count Number of Balanced Permutations"
permalink: "/leetcode/problem/2025-05-09-3343-count-number-of-balanced-permutations/"
leetcode_ui: true
entry_slug: "2025-05-09-3343-count-number-of-balanced-permutations"
---

[3343. Count Number of Balanced Permutations](https://leetcode.com/problems/count-number-of-balanced-permutations/description/) hard
[blog post](https://leetcode.com/problems/count-number-of-balanced-permutations/solutions/6727609/kotlin-rust-by-samoylenkodmitry-870e/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/09052025-3343-count-number-of-balanced?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/rVRN5D3dRsM)
![1.webp](/assets/leetcode_daily_images/259daa1e.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/983

#### Problem TLDR

Permutations of even-odd position sums equal #hard #dp #math

#### Intuition

Didn't solved.
Chain-of-thoughts (mostly irrelevant):

```j
    // 80^80 (will TLE)
    // digits are uniq
    // partition into 2 buckets size/2
    // 123     12       3
    // 12342   132      42
    // i.i.i
    // let's brute force first
    // permutation: can take next any
    // we only have 10 digits: 0,1,2,3,4,5,6,7,8,9
    // we can count them first
    // every even frequency is good to split -- count must be equal
    // every odd frequency should be brute-forced
    // 18 minutes
    // we can calc a sum, then search for a bag of digits to match sum / 2
    // 28 minutes
    // 44 minutes, idea of bugs is not working for 112, a = 2, b = 11
    // the problem with duplicates "11" - not considered a permutation
    // 55 minutes, idea to keep both halves in bags
    // 60 minutes: wrong answer for 53374    4 instead of 6
    // looking for hints:
    // freq (known)
    // dp (somewhat known)
    // useless?
    // 1:15 look for solution
```

Working solution:
* for every digit `i = 9..0`
* take up to `j = frequency[i]` numbers on the one half
* another half would contain `frequency[i] - j` automatically
* sum would change by `i * j` digit times how many we take
* search for the final condition

Now the interesting part - combinatorics. How many permutations we have?
* we take `x` digits and place it at odd positions - that is `Combinations(x, o)`
* and another half `Combinations(frequency[i] - x, e)`

How to count combinations C(a, b)?
* precompute `C[A][B]` like this: `i in 0..a, j in 1..i, c[i][j] = c[i - 1][j] + c[i - 1][j - 1]` (just remember, but better to gain an intuition why it is: permutation is c[i] += c[i - 1], Pascal triangle is computing over previos row, and why it is relevant https://www.perplexity.ai/search/why-combinations-c-i-j-c-i-j-1-5fkL_k8BRXmllKs_FieusA)

#### Approach

* gave up after 1hr
* whats missing: combinatorics intuition

#### Complexity

- Time complexity:
$$O(n^3)$$

- Space complexity:
$$O(n^3)$$

#### Code

```kotlin

    fun countBalancedPermutations(s: String): Int {
        var sum = 0; val f = IntArray(10); for (x in s) { ++f[x - '0']; sum += x - '0' }
        val c = Array(81) { LongArray(81)}; val M = 1000000007L
        for (i in s.indices) { c[i][0] = 1; for (j in 1..i) c[i][j] = (c[i - 1][j] + c[i - 1][j - 1]) % M }
        data class K(val i: Int, val odd: Int, val balance: Int); val dp = HashMap<K, Long>()
        fun dfs(i: Int, o: Int, e: Int, b: Int): Long =  if (o == 0 && e == 0 && b == 0) 1L
            else if (i < 0 || o < 0 || e < 0 || b < 0) 0L else dp.getOrPut(K(i, o, b)) {
                var res = 0L; for (j in 0..f[i]) res +=
                    (((c[o][j] * c[e][f[i] - j]) % M) * dfs(i - 1, o - j, e - f[i] + j, b - i * j)) % M
                res % M
            }
        return if (sum % 2 > 0) 0 else dfs(9, (1 + s.length) / 2, s.length / 2, sum / 2).toInt()
    }

```
```rust

    pub fn count_balanced_permutations(s: String) -> i32 {
        let (mut sum, mut f, mut c, M) = (0, [0; 10], [[0; 81]; 81], 1000000007);
        for b in s.bytes() { f[(b - b'0') as usize] += 1; sum += (b - b'0') as i64 }
        if sum & 1 > 0 { return 0 }; let mut dp = [[[-1; 81]; 81]; 361];
        for i in 0..81 { c[i][0] = 1; for j in 1..=i { c[i][j] = (c[i - 1][j] + c[i - 1][j - 1]) % M }}
        fn dfs(i: i64, o: i64, e: i64, b: i64, c: &[[i64; 81]; 81], dp: &mut [[[i64; 81]; 81]; 361], f: &[i64; 10]) -> i64 {
            if o == 0 && e == 0 && b == 0 { return 1 } else if i.min(o).min(e).min(b) < 0 { return 0 }
            if dp[b as usize][i as usize][o as usize] >= 0 { return dp[b as usize][i as usize][o as usize] };
            let (mut r, M) = (0, 1000000007);
            for j in 0..=f[i as usize] {
                let k = f[i as usize] - j; let comb = (c[o as usize][j as usize] * c[e as usize][k as usize]) % M;
                let next = dfs(i - 1, o - j, e - k, b - i * j, c, dp, f);
                r = (r + (comb * next) % M) % M
            }
            dp[b as usize][i as usize][o as usize] = r; r
        } dfs(9, (1 + s.len() as i64) / 2, s.len() as i64 / 2, sum / 2, &c, &mut dp, &f) as _
    }

```
```c++

    int countBalancedPermutations(string s) {
        int sum = 0, f[10]={}, c[81][81]={}, dp[81][81][361]={}, M = 1e9+7;
        for(auto c: s) { ++f[c - '0']; sum += c - '0'; }; if (sum & 1 > 0) return 0;
        for (int i = 0; i < 81; ++i)
            { c[i][0] = 1; for (int j = 1; j <= i; ++j) c[i][j] = (c[i - 1][j] + c[i - 1][j - 1]) % M; }
        auto d = [&](this const auto& d, int i, int o, int e, int b) -> int {
            if (o == 0 && e == 0 && b == 0) return 1; if (i < 0 || o < 0 || e < 0 || b < 0) return 0;
            if (dp[i][o][b]) return dp[i][o][b] - 1;
            int r = 0;
            for (int j = 0; j <= f[i]; ++j) {
                int k = f[i] - j; int comb = (1LL * c[o][j] * c[e][k]) % M;
                int next = d(i - 1, o - j, e - k, b - i * j);
                r = (1LL * r + (1LL * comb * next) % M) % M;
            }
            dp[i][o][b] = r + 1; return r;
        };
        return d(9, (1 + size(s)) / 2, size(s) / 2, sum / 2);
    }

```

