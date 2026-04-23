---
layout: leetcode-entry
title: "138. Copy List with Random Pointer"
permalink: "/leetcode/problem/2023-09-05-138-copy-list-with-random-pointer/"
leetcode_ui: true
entry_slug: "2023-09-05-138-copy-list-with-random-pointer"
---

[138. Copy List with Random Pointer](https://leetcode.com/problems/copy-list-with-random-pointer/description/) medium
[blog post](https://leetcode.com/problems/copy-list-with-random-pointer/solutions/4003603/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/5092023-138-copy-list-with-random?utm_campaign=post&utm_medium=web)

![image.png](/assets/leetcode_daily_images/a9a44ef4.webp)

#### Problem TLDR

Copy of a graph

#### Intuition

Simple way is just store mapping `old -> new`.
The trick from hint is to store new nodes in between the old ones, then mapping became `old -> new.next & new -> old.next`.

#### Approach

One iteration to make new nodes, second to assign `random` field and final to split lists back.

#### Complexity

- - Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun copyRandomList(node: Node?): Node? {
      var curr = node
      while (curr != null) {
        val next = curr.next
        curr.next = Node(curr.`val`).apply { this.next = next }
        curr = next
      }
      curr = node
      while (curr != null) {
        curr.next?.random = curr.random?.next
        curr = curr.next?.next
      }
      curr = node
      val head = node?.next
      while (curr != null) {
        val currNew = curr.next
        val nextOrig = currNew?.next
        val nextNew = nextOrig?.next
        curr.next = nextOrig
        currNew?.next = nextNew
        curr = nextOrig
      }
      return head
    }

```

