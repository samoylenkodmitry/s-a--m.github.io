---
layout: leetcode-entry
title: "799. Champagne Tower"
permalink: "/leetcode/problem/2026-02-14-799-champagne-tower/"
leetcode_ui: true
entry_slug: "2026-02-14-799-champagne-tower"
---

[799. Champagne Tower](https://leetcode.com/problems/champagne-tower/description/) medium
[blog post](https://leetcode.com/problems/champagne-tower/solutions/7578388/kotlin-rust-by-samoylenkodmitry-obbo/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/14022026-799-champagne-tower?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/dGVSq6mblVA)

![77e66a34-bf79-41a8-83a6-c5e8f1976b59 (1).webp](/assets/leetcode_daily_images/a2c7bd23.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1268

#### Problem TLDR

Flow in a Pascal's Triangle #medium

#### Intuition

```j
    // only 100 rows, we can simulate
    // however p is 10^9, so not one-by-one
    //                             x
    //                       x/2       x/2
    //                   x/4   (x/4+x/4)    x/4
    //            x/8    (x/8+x/4) (x/4+x/8)     x/8
    //    x/16  (x/16+x/16+x/8) (x/8+x/4) (x/8+x/16+x/16)  x/16
    //
    // p=1 only first x filled
    // p=2 2-1-1/2-1/2
    // p=3 3-1-(1/2+1/2+1/2+1/2)
    // p=4 4-1-2-(1/4+1/4+1/4+1/4)
    // how to look at this problem?
    //
    // given 4 cups
    // look at x - minus 1 cup, now have 3 cups
    // look at x/2 - to make it full i need 2 cups, can i have them?
    //               so mark it full and at this row we spend 2 cups
    //               1 cup extra
    // look at x/2 - we spending 2 cups at this row, so the cup is full
    //               and our 1 extra cup stays
    // look at x/4 - to make it full we need 4 cups, but have only 1
    //               so this place is 1/4, and we spend 1 cup at this row
    // look at x/4+x/4 - 1/2
    // x/4 - 1/4
    // next row x/8  0 cups left, ok but what if we have 6 cups initially
    // x - 1 5left
    // x/2 - 1 take2 3left
    // x/2 - 1
    // x/4   3/4 take3 0left
    // (x/4+x/4) 5/4 overflows 0left and 1/4 goes under this cup
    // x/4  3/4
    // x/8 -- 0/8
    // x/8+x/4 5/8 so , should we maintain overflows individually?
    // any better angle to look at this problem?
    // 24 minute
    //            4
    //      3/2        3/2
    //  1/4    1/4+1/4    1/4
    //
```

Simulate the `flow`. Keep the flow values at cells. The next row use the (previouse-1)/2

#### Approach

* use an arena allocation of a single array[r]
* right-aligned pyramid allows to iterate only forward

#### Complexity

- Time complexity:
$$O(r^2)$$

- Space complexity:
$$O(r)$$

#### Code

```kotlin
// 107ms
    fun champagneTower(p: Int, r: Int, c: Int): Double {
        var o = DoubleArray(r+1); o[r] = 1.0*p
        for (j in 1..r) for (i in r-j+1..r)
            o[i] = max(0.0, (o[i]-1.0)/2).also { o[i-1] += it }
        return min(1.0, o[c])
    }
```
```rust
// 1ms
    pub fn champagne_tower(p: i32, r: i32, c: i32) -> f64 {
        let r = r as usize; let mut o = [0.;100]; o[r] = p as f64;
        for j in 1..=r { for i in 1+r-j..=r {
            o[i] = 0.0f64.max(o[i]-1.)/2.; o[i-1] += o[i]
        }} o[c as usize].min(1.)
    }
```

