---
layout: leetcode-entry
title: "3739. Count Subarrays With Majority Element II"
permalink: "/leetcode/problem/2026-06-26-3739-count-subarrays-with-majority-element-ii/"
leetcode_ui: true
entry_slug: "2026-06-26-3739-count-subarrays-with-majority-element-ii"
---

[3739. Count Subarrays With Majority Element II](https://leetcode.com/problems/count-subarrays-with-majority-element-ii/solutions/8359478/kotlin-rust-by-samoylenkodmitry-hsf0/) hard
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/26062026-3739-count-subarrays-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Ej-Equz8M_M)

https://dmitrysamoylenko.com/leetcode/

![26.06.2026.webp](/assets/leetcode_daily_images/26.06.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1402

#### Problem TLDR

Subarrays with majority element

#### Intuition

Didn't solve.
```j
    // 1 2 2 3         2
    //-1 1 1-1   f[0]=1
    //-1         b=-1         f[-1]=1
    //   1       b=0     good f[0]=2     how many prefix sums are bigger than -b (lower than b)
    //     1     b=1     good f[1]=1     actually: how many running sums with exact -b value
    ////    -1   b=0     good f[0]=3     plus ongoing positive streak -- doesnt work
```

1. convert to the balance -1 +1 sequence b = running sum of this
2. keep track of balances frequencies f[b]
3. f[curr_b] - b[j] is the subarray balance, it should be positive if we want majority
4. so for the f[curr_b] we want to know how many j balances we visited with lesser f[j]
5. we can use TreeMap/SegmentTree for that
6. for O(n) track the lesser-balance i points count in a single variable c
7. if balance grows +1, the lesser count is previous lesser count plus exact f[b] of the previous b
8. if balance shrinks -1, the lesser count is the previous lesser count minus new smaller b, f[b-1]

#### Approach

* don't forget sentinel start of the prefix sum

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun countMajoritySubarrays(n: IntArray, t: Int): Long {
        val f = HashMap<Int, Int>(); var b = 0; var c = 0L; f[0] = 1
        return n.sumOf { x ->
            if (x == t) c += f[b++] ?: 0 else c -= f[--b] ?: 0
            f[b] = 1 + (f[b] ?: 0); c
        }
    }
```
```rust
    pub fn count_majority_subarrays(n: Vec<i32>, t: i32) -> i64 {
        let (mut f,mut b, mut c) = (vec![0; n.len()*2+2],0,0); f[n.len()] = 1;
        n.iter().map(|&x| {
            if x == t { c += f[b+n.len()]; b += 1 } else { b -=1; c -= f[b+n.len()] }
            f[b + n.len()] += 1; c
        }).sum()
    }
```

