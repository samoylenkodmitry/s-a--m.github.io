---
layout: leetcode-entry
title: "225. Implement Stack using Queues"
permalink: "/leetcode/problem/2023-08-28-225-implement-stack-using-queues/"
leetcode_ui: true
entry_slug: "2023-08-28-225-implement-stack-using-queues"
---

[225. Implement Stack using Queues](https://leetcode.com/problems/implement-stack-using-queues/description/) easy
[blog post](https://leetcode.com/problems/implement-stack-using-queues/solutions/3969874/kotlin-rotate/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/28082023-225-implement-stack-using?utm_campaign=post&utm_medium=web)

![image.png](/assets/leetcode_daily_images/d907dddb.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/322

#### Problem TLDR

Create a Stack using Queue's push/pop methods.

#### Intuition

We can use a single Queue, and rotate it so that the newly inserted element will be on a first position:

```
1 push -> [1]
2 push -> [1 2] -> [2 1]
3 push -> [2 1 3] -> [1 3 2] -> [3 2 1]
```

#### Approach

Kotlin has no methods `pop`, `push` and `peek` for `ArrayDeque`, use `removeFirst`, `add` and `first`.

#### Complexity

- Time complexity:
$$O(n)$$ for insertions, others are O(1)

- Space complexity:
$$O(n)$$ for internal Queue, and O(1) operations overhead

#### Code

```kotlin
class MyStack: Queue<Int> by LinkedList() {
    fun push(x: Int) {
      add(x)
      repeat(size - 1) { add(pop()) }
    }
    fun pop() = remove()
    fun top() = first()
    fun empty() = isEmpty()
}
```

