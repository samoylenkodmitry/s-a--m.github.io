---
layout: leetcode-entry
title: "3312. Sorted GCD Pair Queries"
permalink: "/leetcode/problem/2026-07-17-3312-sorted-gcd-pair-queries/"
leetcode_ui: true
entry_slug: "2026-07-17-3312-sorted-gcd-pair-queries"
---

[3312. Sorted GCD Pair Queries](https://leetcode.com/problems/sorted-gcd-pair-queries/solutions/8402768/kotlin-rust-by-samoylenkodmitry-gib2/) hard
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/17072026-3312-sorted-gcd-pair-queries?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/oB0oVptzR34)

https://dmitrysamoylenko.com/leetcode/

![17.07.2026.webp](/assets/leetcode_daily_images/17.07.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1423

#### Problem TLDR

Queries of position of gcd of every pair in a sorted order

#### Intuition

Didn't solved
```j
    // 5:30 - have no idea
    //        dp[i] = how many gcds are up to i
    //        sort numbers, 1 2 4 4
    //                          i        ?
    // 8: 00 look for hints: number of pair that have some gcd=g
    //                       again, no idea how
    //       inclusion-exclusion: ?
    // 11:38 gave up
```
Count number of gcds by computing for each gcd how many multipliers we have. Precompute the multipliers with frequency.
Subtract overcounted results of gcd[2*a],[3*a] and so on.
Binary search in a prefix sum of gcds count.

#### Approach

* computing multipliers in a forward way to find all gcds count is a clever idea

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun gcdValues(n: IntArray, q: LongArray) = run {
        val m = n.max(); val c = IntArray(m+1); for (x in n) ++c[x]; val g = LongArray(m+1)
        for (a in m downTo 1) g[a] = (a..m step a)
            .sumOf{c[it]}.let { 1L*it*(it-1)/2}-(a*2..m step a).sumOf {g[it]}
        for (i in 1..m) g[i] += g[i - 1]
        q.map { v -> g.asList().binarySearch { if (it <= v) -1 else 1 }.inv() }
    }
```
```rust
    pub fn gcd_values(n: Vec<i32>, q: Vec<i64>) -> Vec<i32> {
        let m = *n.iter().max().unwrap() as usize; let mut f = vec![0u64; m + 1];
        let mut g = f.clone(); for n in n { f[n as usize] += 1 };
        for i in (1..=m).rev() {
            let k: u64 = (i..=m).step_by(i).map(|j| f[j]).sum();
            g[i] = k * (k - 1) / 2 - (i * 2..=m).step_by(i).map(|j| g[j]).sum::<u64>();
        }
        for i in 1..=m { g[i] += g[i - 1] }
        q.iter().map(|&x| g.partition_point(|&v| v <= x as u64) as i32).collect()
    }
```

