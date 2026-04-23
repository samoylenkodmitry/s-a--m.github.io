---
layout: leetcode-entry
title: "3010. Divide an Array Into Subarrays With Minimum Cost I"
permalink: "/leetcode/problem/2026-02-01-3010-divide-an-array-into-subarrays-with-minimum-cost-i/"
leetcode_ui: true
entry_slug: "2026-02-01-3010-divide-an-array-into-subarrays-with-minimum-cost-i"
---

[3010. Divide an Array Into Subarrays With Minimum Cost I](https://leetcode.com/problems/divide-an-array-into-subarrays-with-minimum-cost-i/description/) easy
[blog post](https://leetcode.com/problems/divide-an-array-into-subarrays-with-minimum-cost-i/solutions/7542464/kotlin-rust-by-samoylenkodmitry-qylv/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/01022026-3010-divide-an-array-into?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/BO7QVtLO2JI)

![8e018255-29fb-4bde-8fd3-fc38dbcc4632 (1).webp](/assets/leetcode_daily_images/503e73ad.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1255

#### Problem TLDR

Split on 2 min values #easy #quickselect

#### Intuition

Array is 50 elements. Sort and take 2. Or scan and deal with ifs.

#### Approach

* bitmask solution is also possible, enabling a branchless vectorizable code

#### Complexity

- Time complexity:
$$O(nlog(n))$$, fastest is O(n)

- Space complexity:
$$O(n)$$, fastest is O(1)

#### Code

```kotlin
// 10ms
    fun minimumCost(n: IntArray) =
    n.run{sort(1);n[0]+n[1]+n[2]}
    /*
    n.run{sort(1);n.take(3).sum()}
    n[0]+n.drop(1).sorted().take(2).sum()
    */
```
```rust
// 0ms
    pub fn minimum_cost(mut n: Vec<i32>) -> i32 {
        n[1..].sort();n[0]+n[1]+n[2]
    }
        /*
        n[1..].select_nth_unstable(1);n[0]+n[1]+n[2]

        n[0] + n[1..].iter().sorted().take(2).sum::<i32>()

        let (a,b)=n[1..].iter().fold((0u64,0),|(a,b),&x|(a|(1<<x),b|a&(1<<x)));
        n[0] + (a.trailing_zeros() + (b | a & a - 1).trailing_zeros()) as i32
        */
```

