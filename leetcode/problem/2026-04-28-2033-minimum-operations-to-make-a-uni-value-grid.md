---
layout: leetcode-entry
title: "2033. Minimum Operations to Make a Uni-Value Grid"
permalink: "/leetcode/problem/2026-04-28-2033-minimum-operations-to-make-a-uni-value-grid/"
leetcode_ui: true
entry_slug: "2026-04-28-2033-minimum-operations-to-make-a-uni-value-grid"
---

[2033. Minimum Operations to Make a Uni-Value Grid](https://leetcode.com/problems/minimum-operations-to-make-a-uni-value-grid/solutions/8107163/kotlin-rust-by-samoylenkodmitry-o9rt/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/28042026-2033-minimum-operations?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/HCM-GNjUi-g)

https://dmitrysamoylenko.com/leetcode/

![28.04.2026.webp](/assets/leetcode_daily_images/28.04.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1341

#### Problem TLDR

Min ops to make all numbers equal by +-X

#### Intuition

```j
    // 1 4       x=3
    //+3
    //
    // 1 8 8 8 8  x=7
    //
    // is the baseline binary searchable?
    //
    // 1 1 3 5 5   x=2
    // 2 2 1 0 0
    // 1 1 0 1 1
    // 0 0 1 2 2
    //
    // 1 1 1 1 1    %2 same reminder
    // subtract reminder
    // 0 0 2 4 4, then divide by x
    // 0 0 1 2 2  this is array of counts from zero
    //            now find a median?
    // 2 4 6 8   x=2
    // 1 2 3 4 sum=10
    //          maybe consider each item as base and check
    // *
    // 1 *
    // 2 1 *
    // 3 2 1 *
    // 2 1 * 1
    // 1 * 1 2
    // * 1 2 3
    //
    // 1 4 4 5 8
    // * 3 3 4 7  right=3+3+4+7=17 left = 0
    // 3 * 0 1 4  right=  0+1+4=5=17-4*(4-1) left=0+(4-1)=3
    // 3 0 * 1 4  right= 5-3*(0-0) left=3+(0-0)
    // 4 1 1 * 3  right= 5-2*(5-4)=3 left=3+3*(5-4)=6
    // 7 4 4 3 *  right= 3-1*(8-5)=0 left=6+4*(8-5)=18
    //
```

1. baseline of each value sequence is value %X
2. number of ops is (V - V%X)/X
3. scan from left to right, see how number of ops to the right changes after each move

#### Approach

* if you know math, just use median, it is the middle of the array
* kotlin&rust has a cool way to flatten grid

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(nm)$$

#### Code

```kotlin
    fun minOperations(g: Array<IntArray>, v: Int) = g.reduce{a,b->a+b}
    .sorted().run{sumOf {if((it-get(0))%v>0)return-1;abs(it-get(size/2))/v}}
```
```rust
    pub fn min_operations(g: Vec<Vec<i32>>, x: i32) -> i32 {
        let mut a=g.concat();a.sort();let m=a[a.len()/2];
        if a.iter().any(|v|(v-a[0])%x!=0){-1}else{a.iter().map(|v|(v-m).abs()/x).sum()}
    }
```

