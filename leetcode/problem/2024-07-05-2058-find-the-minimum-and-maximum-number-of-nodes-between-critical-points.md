---
layout: leetcode-entry
title: "2058. Find the Minimum and Maximum Number of Nodes Between Critical Points"
permalink: "/leetcode/problem/2024-07-05-2058-find-the-minimum-and-maximum-number-of-nodes-between-critical-points/"
leetcode_ui: true
entry_slug: "2024-07-05-2058-find-the-minimum-and-maximum-number-of-nodes-between-critical-points"
---

[2058. Find the Minimum and Maximum Number of Nodes Between Critical Points](https://leetcode.com/problems/find-the-minimum-and-maximum-number-of-nodes-between-critical-points/description/) medium
[blog post](https://leetcode.com/problems/find-the-minimum-and-maximum-number-of-nodes-between-critical-points/solutions/5419507/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/5072024-2058-find-the-minimum-and?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Q3X4ECBK9JQ)
![2024-07-05_09-00_1.webp](/assets/leetcode_daily_images/9b9d260e.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/660

#### Problem TLDR

[min, max] distance between critical points in linked list #medium #linked_list

#### Intuition

Just do what is asked.

#### Approach

* we can reuse previous variables `a` and `b`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun nodesBetweenCriticalPoints(head: ListNode?): IntArray {
        var first = -1; var last = -1; var min = Int.MAX_VALUE
        var i = 0; var curr = head?.next; val e = intArrayOf(-1, -1)
        var a = head?.`val` ?: return e; var b = curr?.`val` ?: return e
        while (curr?.next != null) {
            if (a > b && b < curr.next.`val` || a < b && b > curr.next.`val`)
                if (first == -1) first = i else {
                    min = min(min, i - max(first, last))
                    last = i
                }
            i++; a = b; b = curr.next.`val`; curr = curr.next
        }
        return if (last > 0) intArrayOf(min, last - first) else e
    }

```
```rust

    pub fn nodes_between_critical_points(head: Option<Box<ListNode>>) -> Vec<i32> {
        let (mut first, mut last, mut min, mut i, e) = (-1, -1, i32::MAX, 0, vec![-1, -1]);
        let Some(head) = head else { return e }; let mut a = head.val;
        let Some(mut curr) = head.next else { return e }; let mut b = curr.val;
        while let Some(next) = curr.next {
            if a > b && b < next.val || a < b && b > next.val {
                if first == -1 { first = i } else {
                    min = min.min(i - last.max(first));
                    last = i
                }
            }
            i += 1; a = b; b = next.val; curr = next
        }
        if last > 0 { vec![min, last - first] } else { e }
    }

```

