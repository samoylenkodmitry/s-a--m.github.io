---
layout: leetcode-entry
title: "2615. Sum of Distances"
permalink: "/leetcode/problem/2026-04-23-2615-sum-of-distances/"
leetcode_ui: true
entry_slug: "2026-04-23-2615-sum-of-distances"
---

[2615. Sum of Distances](https://leetcode.com/problems/sum-of-distances/solutions/8069058/kotlin-rust-by-samoylenkodmitry-9mqf/) medium
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/23042026-2615-sum-of-distances?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/848sypoVAUs)

![23.04.2026.webp](/assets/leetcode_daily_images/23.04.2026.webp)
#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1337

#### Problem TLDR

Sum of distances to each occurance

#### Intuition

Didn't solved without a hint.

```j
    // 0123456789
    // a...a....a...
    // i   j    j    4+9
    // j   i    j    4+5
    // j   j    i    9+5
    //
    // a b c d
    // *       |a-b|+|a-c|+|a-d|=b-a + b-a+c-b + b-a+c-b+d-c=-3a+b+c+d
    //   *     |a-b|+|b-c|+|b-d|=b-a + c-b + c-b+d-c=-a-b+c+d
    //     *   |a-c|+|b-c|+|c-d|=b-a+c-b + c-b + d-c=-a-b+c+d
    //       * |a-d|+|b-d|+|c-d|= b-a+c-b+d-c + c-b+d-c + d-c=-a-b-c+3d
    // any way to shortcut this?
    //
    // a b c d e f
    //           *                        5f-e-d-c-b-a sum+4f-2e-2d-2c-2b-2a -sum+6f
    //         *   f-e + 4e-d-c-b-a     = f+3e-d-c-b-a sum+2e-2d-2c-2b-2a
    //       *     f-d+e-d + 3d-c-b-a   = f+e+d -c-b-a sum-2c-2b-2a
    //     *       f-c+e-c+d-c + 2c-b-a = f+e+d -c-b-a sum-2c-2b-2a
    //   *         f-b+e-b+d-b+c-b+b-a  = f+e+d+c-3b-a sum-4b-2a
    // *           f-a+e-a+d-a+c-a+b-a  = f+e+d+c+b-5a sum -6a
    // ok what the rule?
    // (acceptance rate 40%)
    // from hints: freq*idx-sum
    // a b c  d
    //  +x=prev+x
    //  +x+2y=prev+2y
    //  +x+2y+3z=prev+3z
```
The simple working intuition:
* consider only forward pass for now
* each position sum is the previous positions sum plus frequency so-far count of distances to last occurence

#### Approach

* we can iterate once but with two cursors: forward and backward and duplicate pairs of variables have to be tracked

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
    fun distance(n: IntArray) = LongArray(n.size).apply {
        val m = Array(2) { HashMap<Int, LongArray>() }
        for (i in n.indices) for ((j,s) in listOf(i to 2, (size - 1 - i) to 0))
            m[s/2].getOrPut(n[j]) { LongArray(2) }.let { a ->
                this[j] += (s-1) * (j * a[0] - a[1]); a[0]++; a[1] += 1L*j
            }
    }
```
```rust
    pub fn distance(n: Vec<i32>) -> Vec<i64> {
        let (mut r, mut m) = (vec![0; n.len()], HashMap::new());
        for i in 0..n.len() { for (j, d,s) in [(i, 1,1), (n.len() - 1 - i, 0,-1)] {
            let e = m.entry(n[j]).or_insert([(0, 0); 2]);
            r[j] += s * (j as i64 * e[d].0 - e[d].1);
            e[d].0 += 1; e[d].1 += j as i64;
        }} r
    }
```

