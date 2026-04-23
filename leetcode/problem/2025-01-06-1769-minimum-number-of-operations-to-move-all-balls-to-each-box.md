---
layout: leetcode-entry
title: "1769. Minimum Number of Operations to Move All Balls to Each Box"
permalink: "/leetcode/problem/2025-01-06-1769-minimum-number-of-operations-to-move-all-balls-to-each-box/"
leetcode_ui: true
entry_slug: "2025-01-06-1769-minimum-number-of-operations-to-move-all-balls-to-each-box"
---

[1769. Minimum Number of Operations to Move All Balls to Each Box](https://leetcode.com/problems/minimum-number-of-operations-to-move-all-balls-to-each-box/description/) medium
[blog post](https://leetcode.com/problems/minimum-number-of-operations-to-move-all-balls-to-each-box/solutions/6238518/kotlin-rust-by-samoylenkodmitry-3kow/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/06012025-1769-minimum-number-of-operations?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/mwjyJqmryZI)
[deep-dive](https://notebooklm.google.com/notebook/b319b6a1-894f-477f-bd21-f1a18a28b354/audio)
![1.webp](/assets/leetcode_daily_images/0e07017a.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/856

#### Problem TLDR

Sum distances to all `1` for every position #medium #prefix_sum

#### Intuition

Let's observe an example:

```j

    // 012345
    // 001011
    // * 2 45  2+4+5=11, right = 3
    //  *1 34  11-3=8, right = 3
    //   * 23  8-3=5 , left = 1, right = 2
    //    *12  5-2=3, +1=4
    //     *   3-2=1, +2=3, right = 1, left = 2
    //      *  1-1, 2+2=4

```

* the minimum operations of moving all `1` to position `i` is the sum of the distances
* we can reuse the previous position result: all `1`'s to the right became closer, and all `1`'s to the left increase distance, so we do `sum[i + 1] = sum[i] - right_ones + left_ones`

#### Approach

* we don't need a separate variable for the `left` and `right`, as we always operate on the `balance = left - right`
* careful with the operations order
* single-pass is impossible, as we should know the balance on the first position already

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun minOperations(boxes: String): IntArray {
        var b = 0; var s = 0
        for ((i, c) in boxes.withIndex())
            if (c > '0') { b--; s += i }
        return IntArray(boxes.length) { i ->
            s.also { b += 2 * (boxes[i] - '0'); s += b }
        }
    }

```
```rust

    pub fn min_operations(boxes: String) -> Vec<i32> {
        let (mut b, mut s) = (0, 0);
        for (i, c) in boxes.bytes().enumerate() {
            if c > b'0' { b -= 1; s += i as i32 }
        }
        boxes.bytes().enumerate().map(|(i, c)| {
            let r = s; b += 2 * (c - b'0') as i32; s += b; r
        }).collect()
    }

```
```c++

    vector<int> minOperations(string boxes) {
        int b = 0, s = 0; vector<int> r(boxes.size());
        for (int i = 0; i < boxes.size(); ++i)
            if (boxes[i] > '0') b--, s += i;
        for (int i = 0; i < boxes.size(); ++i)
            r[i] = s, b += 2 * (boxes[i] - '0'), s += b;
        return r;
    }

```

