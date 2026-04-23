---
layout: leetcode-entry
title: "1590. Make Sum Divisible by P"
permalink: "/leetcode/problem/2025-11-30-1590-make-sum-divisible-by-p/"
leetcode_ui: true
entry_slug: "2025-11-30-1590-make-sum-divisible-by-p"
---

[1590. Make Sum Divisible by P](https://leetcode.com/problems/make-sum-divisible-by-p/description/) medium
[blog post](https://leetcode.com/problems/make-sum-divisible-by-p/solutions/7383464/kotlin-rust-by-samoylenkodmitry-y33u/)
[substack](https://dmitriisamoilenko.substack.com/publish/posts/detail/180309618/share-center)
[youtube](https://youtu.be/7AMTdK1wFnk)

![8fd0e802-9119-468e-a321-dda00fc6be1e (1).webp](/assets/leetcode_daily_images/92941cdc.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1189

#### Problem TLDR

Min removal to make sum%p #medium #hashmap

#### Intuition

```j
   // 3 1 4 2       sum=10 p=6
    // idea: prefix sums
    // 3 4 8 10
    // idea: all subarrays ending at position
    // 3 1 4 2
    //   *          1, 3+1=4
    // sum = remove + stay, where stay % p == 0
    // stay_a .. remove .. stay_b
    // stay_a + stay_b % p == 0
    // (prefix_sum + suffix_sum) % p == 0
    // and we want them as big as possible
    //
    // 3 1 4 2
    // i     j     3+2
    //   i   j     3+1+2=6
    //   or
    // i   j       3+4+2=9 we don't know which pointer to move
    //                     is this dp?
    //
    // idea: invert the problem
    // subarray should be divisible by %(sum%p)
    // check
    // 6 3 5 2    p=9    sum=16, 16%9=7, subarray %7
    //     [..]
    // ok seems works; how can we find it?
    // prefix sum
    //
    // 6 9 14 16
    // reminder to position
    // 6 2 0  2
    //        i    6-0, 2-1, 0-2, currSum=16, currRem=2 (lookup last pos of 2)
    //
    // check
    // 3 1 4 2    sum=10, p=6, sum%p = 4
    // 3 4 8 10
    // 3 0 0 2
    //   i        len=2
    //     i      len=1
    //
    // ok failed on 4 4 2 p=7
    //
    // 4 4 2
    // 4 8 10     k=10%7=3
    // 1 2 1      looks like it completely not working; we want to find %3 subarray
    //            so, 4 2 is %3 but the leftover 4 is not %7, looks like wrong intuition
    //            maybe we want to find at most 3, 10-7=3
    // another fail
    // 3 6 8 1 p=8    sum=18   k=18%8=2
    // 31 minute
    // 1 1 1 0      remove 6 because it is %2, looks like a wrong idea
    //
    // 34 minute go for hints: same ideas as mine, but put s%p instead of s%k in map
    // looks like i overcomplicated and got lost
    // 3 1 4 2     p=6
    // 3 4 2 4
    //
    // 6 3 5 2    p=9 k=7
    // 6 2 0 2
    //
    // so the wrong part is that k should match exactly, not by %k
```

(pref_i - pref_j) % p = k
pref_j %p = pref_i % p - k

#### Approach

* store and lookup pref % p

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 55ms
    fun minSubarray(n: IntArray, p: Int): Int {
        val m = HashMap<Int, Int>(); m[0] = -1; var s = 0
        val k = n.fold(0) {r,t->(r+t)%p}
        return if (k == 0) 0 else n.indices.minOf { i ->
            s = (s + n[i])%p; val j = m[(s-k+p)%p]; m[s] = i
            i - (j?:-n.size)
        }.takeIf { it < n.size } ?: -1
    }
```
```rust
// 15ms
    pub fn min_subarray(n: Vec<i32>, p: i32) -> i32 {
        let k = n.iter().fold(0, |r, &t| (r + t) % p); if k < 1 { return 0 }
        let (mut m, mut s) = (HashMap::from([(0,-1)]), 0);
        n.iter().enumerate().map(|(i, &x)| { s = (s + x) % p;
            let d = m.get(&((s-k+p)%p)).map_or(n.len() as i32, |&j|i as i32 - j);
            m.insert(s, i as i32); d
        }).min().filter(|&r|r < n.len() as i32).unwrap_or(-1)
    }
```

