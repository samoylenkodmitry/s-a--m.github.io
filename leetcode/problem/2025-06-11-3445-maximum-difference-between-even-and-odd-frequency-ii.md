---
layout: leetcode-entry
title: "3445. Maximum Difference Between Even and Odd Frequency II"
permalink: "/leetcode/problem/2025-06-11-3445-maximum-difference-between-even-and-odd-frequency-ii/"
leetcode_ui: true
entry_slug: "2025-06-11-3445-maximum-difference-between-even-and-odd-frequency-ii"
---

[3445. Maximum Difference Between Even and Odd Frequency II](https://leetcode.com/problems/maximum-difference-between-even-and-odd-frequency-ii/description) hard
[blog post](https://leetcode.com/problems/maximum-difference-between-even-and-odd-frequency-ii/solutions/6832054/kotlin-rust-by-samoylenkodmitry-pjp0/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/11062025-3445-maximum-difference?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/x7qrLe9wAYo)
![1.webp](/assets/leetcode_daily_images/e3ad3677.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1016

#### Problem TLDR

Max odd - min even frequency, window at least k #hard

#### Intuition

Didn't solve.

Irrelevant chain-of-thoughts:
```j
    // 0 1 2 3 4   odd-even
    //
    // 1111122    at least k
    //         3-2 k=6
    // small number of digits, maybe solve for each pair
    // even should be the smallest = 2
    // odd should be the largest BUT substring length >= k (so can't shrink less than k to make odd)
    // for every two numbers ..a...a... find the best start..end for odds
    // this is O(n^2) algo, as the window size k up to n
    // can we do it in a one go?
    // 112211221122
    //     eeo        for example consider those even positions
    //    oee         odds with different lengths
    //   ooeeo
    //    oeeoo
    //   ooeeooeeo adds two more evens, no gaps allowed
    //    oeeooeeoo adds two more evens, no gaps allowed
    //                                                                  (17 minute)
    //      eooeo
    // can optimal evens be more than 2?
    // 111112222111111    5-2=3 vs 11-4=5 yes
    // 5    2 2 6
    // ok, maybe FIX the window size and binary search it? - will not work, no criteria for binary search
    //                                                                   (26 minute, 0 lines of code)
    // idea: the only reason to shrink window is to make odd from even   (29 minute)
    //       or, maybe to decrease even frequency                        (33 minute)
    // ok, look for hints, no working ideas yet
    // hint1: fix 2 chars (kind of was close)
    // hint2: prefix sum
    //
    // 111111222211111    5-2=3 vs 11-4=5 yes
    // 6     2 2 5
    //             but how to use the prefix sum?                        (56 minute)
    // (60 minute, give up look for solution)
    // a odd b odd  complementary to   a odd  b even, a even b odd
    // a odd b even complementary to   a even b even, a odd b odd
    // a even b odd complementary to   a even b even, a odd b odd
    // a even b even complementary to  a odd b even, a even b odd  (but what about k?, 75 minute)
```

The missing part for me even after hints was `how to shrink the window`. Basically, we moving until last a or b, `shrinking to size of 2`: aa or bb but preserving at least k.

The working solution:
* for every pair of digits a and b
* compute prefix sum of frequencies fa, fb
* and maintain sliding window with left pointer j
* move j while window at least k and until fa[j] == fa || fb[j] == fb (until last a or b)
* compute diff and put to seen[key]=diff, key is a mask of parity (a%2, b%2)
* then the current complementary key is inversion of parity (1 - fa%2)
* and diff = diff - complementary_diff

#### Approach

* felt close, but not enough
* the rule for shrinking window is crucial here; why shrink to size of 2 gives optimal, and don't make worse other pointer expansion?
* the initial condition to prefixes array is tricky, use sentinel 0 at start
* rotate a b and b a to simplify complementary state matching
* store the `diff=fa[j]-fb[j]` in seen instead of indices, and we have to subtract it as complementary
* or we can skip the prefix array entirely, as we only interested in the latest count

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

// 54ms
    fun maxDifference(s: String, k: Int): Int {
        var res = -s.length
        for (a in "01234") for (b in "01234") if (a != b) {
            val fa = IntArray(s.length + 1); val fb = IntArray(s.length + 1)
            val seen = IntArray(4) { s.length }; var j = 0
            for ((i, c) in s.withIndex()) {
                val l = i + 1; fa[l] = fa[i]; fb[l] = fb[i]
                if (c == a) ++fa[l]; if (c == b) ++fb[l]
                while (j <= i - k + 1 && fb[j] < fb[l]) {
                    val key = (fa[j] % 2) * 2 + (fb[j] % 2)
                    seen[key] = min(fa[j] - fb[j], seen[key])
                    j++
                }
                res = max(res, fa[l] - fb[l] - seen[(1 - fa[l] % 2) * 2 + (fb[l] % 2)])
            }
        }
        return res
    }

```
```rust

// 26ms
    pub fn max_difference(s: String, k: i32) -> i32 {
        let (k, s, n, mut r) = (k as usize, s.as_bytes(), s.len(), -(s.len() as i32));
        for &a in b"01234" { for &b in b"01234" { if a == b { continue; }
            let (mut fa, mut pa, mut fb, mut pb, mut seen, mut j) =
                (0, 0, 0, 0, vec![n as i32; 4], 0);
            for (i, &c) in s.iter().enumerate() {
                fa += (c == a) as i32; fb += (c == b) as i32;
                while j + k <= i + 1 && fb >= 2 + pb {
                    let key = ((pa % 2) * 2 + (pb % 2)) as usize;
                    seen[key] = seen[key].min(pa - pb);
                    pa += (s[j] == a) as i32; pb += (s[j] == b) as i32;
                    j += 1;
                }
                r = r.max(fa - fb - seen[((1 - fa % 2) * 2 + (fb % 2)) as usize]);
        }}} r
    }

```
```c++

// 39ms
    int maxDifference(string s, int k) {
        int n = s.size(), r = -n;
        for (char a : {'0','1','2','3','4'})
        for (char b : {'0','1','2','3','4'}) if (a != b) {
            int fa = 0, fb = 0, pa = 0, pb = 0, j = 0, seen[4] = {n,n,n,n};
            for (int i = 0; i < n; ++i) {
                fa += s[i] == a; fb += s[i] == b;
                while (j + k <= i + 1 && fb >= pb + 2) {
                    int key = pa % 2 * 2 + pb % 2;
                    seen[key] = min(seen[key], pa - pb);
                    pa += s[j] == a; pb += s[j] == b; ++j;
                }
                int key = (1 - fa % 2) * 2 + fb % 2;
                r = max(r, fa - fb - seen[key]);
            }
        } return r;
    }

```

