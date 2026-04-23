---
layout: leetcode-entry
title: "1171. Remove Zero Sum Consecutive Nodes from Linked List"
permalink: "/leetcode/problem/2024-03-12-1171-remove-zero-sum-consecutive-nodes-from-linked-list/"
leetcode_ui: true
entry_slug: "2024-03-12-1171-remove-zero-sum-consecutive-nodes-from-linked-list"
---

[1171. Remove Zero Sum Consecutive Nodes from Linked List](https://leetcode.com/problems/remove-zero-sum-consecutive-nodes-from-linked-list/description/) medium
[blog post](https://leetcode.com/problems/remove-zero-sum-consecutive-nodes-from-linked-list/solutions/4863090/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/12032024-1171-remove-zero-sum-consecutive?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/P4GnJouHViM)
![2024-03-12_10-03.jpg](/assets/leetcode_daily_images/ef2638da.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/536

#### Problem TLDR

Remove consequent 0-sum items from a LinkedList #medium

#### Intuition
Let's calculate running sum and check if we saw it before.
The corner case example:
```j
    // 1 3 2 -3 -2 5 5 -5 1
    // 1 4 6  3  1 6 11 6 7
    //   - -  -  -
    //     x         -  -
    // 1           5      1
```
We want to remove `3 2 -3 -2` but `sum = 6` is yet stored in our HashMap. So we need to manually clean it. This will not increse the O(n) time complexity as we are walk at most twice.

#### Approach

The Rust approach is O(n^2). We operate with references like this: first `.take()` then `insert(v)` back. (solution from https://leetcode.com/discreaminant2809/)

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

  fun removeZeroSumSublists(head: ListNode?): ListNode? {
    val dummy = ListNode(0).apply { next = head }
    val sumToNode = mutableMapOf<Int, ListNode>()
    var n: ListNode? = dummy; var sum = 0
    while (n != null) {
      sum += n.`val`
      val prev = sumToNode[sum]
      if (prev != null) {
        var x: ListNode? = prev.next
        var s = sum
        while (x != n && x != null) {
          s += x.`val`
          if (x == sumToNode[s]) sumToNode.remove(s)
          x = x.next
        }
        prev.next = n.next
      } else sumToNode[sum] = n
      n = n.next
    }
    return dummy.next
  }

```
```rust

    pub fn remove_zero_sum_sublists(mut head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
      let mut node_i_ref = &mut head;
      'out: while let Some(mut node_i) = node_i_ref.take() {
        let (mut node_j_ref, mut sum) = (&mut node_i, 0);
        loop {
          sum += node_j_ref.val;
          if sum == 0 {
            *node_i_ref = node_j_ref.next.take();
            continue 'out;
          }
          let Some (ref mut next_node_j_ref) = node_j_ref.next else { break };
          node_j_ref = next_node_j_ref;
        }
        node_i_ref = &mut node_i_ref.insert(node_i).next;
      }
      head
    }

```

